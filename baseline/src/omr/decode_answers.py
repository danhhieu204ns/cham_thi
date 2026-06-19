"""Rule-based answer decoding from bubble state records."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable


CHOICE_ORDER = ("A", "B", "C", "D")
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


def _sort_choice_key(record: dict) -> tuple[int, str]:
    choice = str(record["choice"])
    try:
        return CHOICE_ORDER.index(choice), choice
    except ValueError:
        return len(CHOICE_ORDER), choice


def _state_from_record(record: dict) -> dict:
    state = {
        "prelabel": record["prelabel"],
        "darkness_score": record["darkness_score"],
        "crop_path": record["crop_path"],
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


def _record_score(record: dict | None) -> float:
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
    top_score = max(_record_score(record) for record in records)
    filled_threshold = _as_float(records[0].get("group_filled_threshold"), 0.08) or 0.08
    return round(_clamp(1.0 - (top_score / max(filled_threshold, 1e-6))), 6)


def _decode_confidence(
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


def _alignment_gate(records: Iterable[dict]) -> tuple[bool, bool]:
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


def decode_question(records: Iterable[dict]) -> dict:
    bubbles = sorted(records, key=_sort_choice_key)
    if not bubbles:
        raise ValueError("cannot decode an empty question")

    question_id = str(bubbles[0]["question_id"])
    question_number = int(bubbles[0]["question_number"])
    states = {str(record["choice"]): _state_from_record(record) for record in bubbles}

    filled = [record for record in bubbles if record["prelabel"] == "filled"]
    ambiguous = [record for record in bubbles if record["prelabel"] == "ambiguous"]
    invalid = [record for record in bubbles if record["prelabel"] == "invalid"]
    review_reasons: list[str] = []

    if len(states) != len(CHOICE_ORDER):
        review_reasons.append("incomplete_choices")
    if invalid:
        review_reasons.append("invalid_bubble")
    if ambiguous:
        review_reasons.append("ambiguous_bubble")
    if len(filled) > 1:
        review_reasons.append("multi_mark")
    alignment_needs_review, hard_alignment_failure = _alignment_gate(bubbles)
    if alignment_needs_review:
        review_reasons.append("alignment_failed")

    filled = sorted(filled, key=_record_score, reverse=True)
    selected_record = None
    if not hard_alignment_failure and len(filled) == 1:
        selected_record = filled[0]
    selected = str(selected_record["choice"]) if selected_record is not None else None

    selected_score = _record_score(selected_record) if selected_record is not None else None
    other_scores = [
        _record_score(record)
        for record in bubbles
        if selected is None or str(record["choice"]) != selected
    ]
    score_margin = None
    if selected_score is not None and other_scores:
        score_margin = round(selected_score - max(other_scores), 6)
    confidence = _decode_confidence(
        selected_record,
        selected_score=selected_score,
        score_margin=score_margin,
        records=bubbles,
    )
    if selected is not None and confidence is not None and confidence < 0.45:
        review_reasons.append("low_confidence")

    if review_reasons:
        status = "multi_mark" if "multi_mark" in review_reasons else "need_review"
    elif selected is not None:
        status = "accepted"
    else:
        status = "blank"

    selected_darkness_score = (
        float(selected_record["darkness_score"]) if selected_record is not None else None
    )
    other_darkness_scores = [
        float(record["darkness_score"])
        for record in bubbles
        if selected is None or str(record["choice"]) != selected
    ]
    darkness_margin = None
    if selected_darkness_score is not None and other_darkness_scores:
        darkness_margin = round(selected_darkness_score - max(other_darkness_scores), 6)

    return {
        "question_id": question_id,
        "question_number": question_number,
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


def decode_part1(crop_records: Iterable[dict]) -> list[dict]:
    sheets: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    sheet_meta: dict[str, dict] = {}

    for record in crop_records:
        image_id = str(record["image_id"])
        question_id = str(record["question_id"])
        sheets[image_id][question_id].append(record)
        sheet_meta.setdefault(
            image_id,
            {
                "image_id": image_id,
                "source_path": record.get("source_path"),
                "input_path": record.get("input_path"),
            },
        )

    decoded_sheets: list[dict] = []
    for image_id in sorted(sheets):
        answers = {}
        review_items = []
        status_counts: Counter[str] = Counter()

        for question_id, records in sorted(
            sheets[image_id].items(),
            key=lambda item: int(item[1][0]["question_number"]),
        ):
            question_result = decode_question(records)
            answers[question_id] = question_result
            status_counts[question_result["status"]] += 1

            if question_result["need_review"]:
                review_items.append(
                    {
                        "question": question_id,
                        "question_number": question_result["question_number"],
                        "selected": question_result["selected"],
                        "reasons": question_result["review_reasons"],
                    }
                )

        decoded_sheets.append(
            {
                **sheet_meta[image_id],
                "part": "I",
                "answers": answers,
                "review_items": review_items,
                "counts": dict(sorted(status_counts.items())),
            }
        )

    return decoded_sheets


def summarize_decoded_sheets(decoded_sheets: Iterable[dict]) -> dict:
    sheet_count = 0
    question_count = 0
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    sheets_with_review = 0
    review_item_count = 0

    for sheet in decoded_sheets:
        sheet_count += 1
        answers = sheet["answers"]
        review_items = sheet["review_items"]
        question_count += len(answers)
        sheets_with_review += int(bool(review_items))
        review_item_count += len(review_items)

        for answer in answers.values():
            status_counts[answer["status"]] += 1
            for reason in answer["review_reasons"]:
                reason_counts[reason] += 1

    return {
        "sheet_count": sheet_count,
        "question_count": question_count,
        "status_counts": dict(sorted(status_counts.items())),
        "review_item_count": review_item_count,
        "sheets_with_review": sheets_with_review,
        "review_reason_counts": dict(sorted(reason_counts.items())),
    }
