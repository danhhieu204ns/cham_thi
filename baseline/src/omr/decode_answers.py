"""Rule-based answer decoding from bubble state records."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable


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
    states = {
        str(record["choice"]): {
            "prelabel": record["prelabel"],
            "darkness_score": record["darkness_score"],
            "crop_path": record["crop_path"],
        }
        for record in bubbles
    }

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

    selected = str(filled[0]["choice"]) if len(filled) == 1 else None
    if review_reasons:
        status = "multi_mark" if "multi_mark" in review_reasons else "need_review"
    elif selected is not None:
        status = "accepted"
    else:
        status = "blank"

    filled_scores = [float(record["darkness_score"]) for record in filled]
    selected_score = filled_scores[0] if selected is not None else None
    other_scores = [
        float(record["darkness_score"])
        for record in bubbles
        if selected is None or str(record["choice"]) != selected
    ]
    score_margin = None
    if selected_score is not None and other_scores:
        score_margin = round(selected_score - max(other_scores), 6)

    return {
        "question_id": question_id,
        "question_number": question_number,
        "selected": selected,
        "status": status,
        "need_review": bool(review_reasons),
        "review_reasons": review_reasons,
        "confidence": None,
        "darkness_score": round(selected_score, 6) if selected_score is not None else None,
        "darkness_margin": score_margin,
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
