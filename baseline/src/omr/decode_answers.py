"""Rule-based answer decoding from bubble state records."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from .section_decode import alignment_gate, decode_confidence, record_score, state_from_record


CHOICE_ORDER = ("A", "B", "C", "D")


def _sort_choice_key(record: dict) -> tuple[int, str]:
    choice = str(record["choice"])
    try:
        return CHOICE_ORDER.index(choice), choice
    except ValueError:
        return len(CHOICE_ORDER), choice


def decode_question(records: Iterable[dict]) -> dict:
    bubbles = sorted(records, key=_sort_choice_key)
    if not bubbles:
        raise ValueError("cannot decode an empty question")

    question_id = str(bubbles[0]["question_id"])
    question_number = int(bubbles[0]["question_number"])
    states = {str(record["choice"]): state_from_record(record) for record in bubbles}

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
    alignment_needs_review, hard_alignment_failure = alignment_gate(bubbles)
    if alignment_needs_review:
        review_reasons.append("alignment_failed")

    filled = sorted(filled, key=record_score, reverse=True)
    selected_record = None
    if not hard_alignment_failure and len(filled) == 1:
        selected_record = filled[0]
    selected = str(selected_record["choice"]) if selected_record is not None else None

    selected_score = record_score(selected_record) if selected_record is not None else None
    other_scores = [
        record_score(record)
        for record in bubbles
        if selected is None or str(record["choice"]) != selected
    ]
    score_margin = None
    if selected_score is not None and other_scores:
        score_margin = round(selected_score - max(other_scores), 6)
    confidence = decode_confidence(
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


