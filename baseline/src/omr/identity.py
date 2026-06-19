"""Shared helpers for student-id and exam-code grids."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .geometry import bubble_bbox
from .section_decode import decode_group


def crop_output_path(output_dir: Path, section: str, prelabel: str, crop_name: str) -> Path:
    return output_dir / section / prelabel / crop_name


def build_identity_specs(template: dict, field: str) -> list[dict]:
    grid = template["grids"][field]
    size = tuple(template["bubble_crop"]["size"])
    specs = []
    for column_index, center_x_raw in enumerate(grid["column_x"], start=1):
        center_x = int(round(center_x_raw))
        for digit_index, digit in enumerate(grid["digits"]):
            center_y = int(round(float(grid["row_y_start"]) + digit_index * float(grid["row_y_step"])))
            specs.append(
                {
                    "section": "identity",
                    "field": field,
                    "spec_id": f"{field}_{column_index:02d}_{digit}",
                    "item_id": f"{field}_{column_index:02d}",
                    "question_id": f"{field}_{column_index:02d}",
                    "question_number": column_index,
                    "choice": digit,
                    "label": digit,
                    "role": "digit",
                    "slot": column_index,
                    "center": [center_x, center_y],
                    "bbox": list(bubble_bbox(center_x, center_y, size)),
                }
            )
    return specs


def relabel_identity_groups(
    crop_records: list[dict],
    *,
    filled_threshold: float,
    margin_threshold: float,
    score_key: str = "darkness_score",
    output_dir: Path | None = None,
    project_root: Path | None = None,
) -> None:
    """Classify digit columns by rank so printed digits do not become marks."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in crop_records:
        if record["section"] == "identity":
            groups[(record["image_id"], record["item_id"])].append(record)

    for records in groups.values():
        records.sort(key=lambda record: float(record.get(score_key) or 0.0), reverse=True)
        if not records:
            continue

        top_score = float(records[0].get(score_key) or 0.0)
        second_score = float(records[1].get(score_key) or 0.0) if len(records) > 1 else 0.0
        if top_score < filled_threshold:
            labels = {record["spec_id"]: "blank" for record in records}
        elif top_score - second_score < margin_threshold:
            labels = {
                record["spec_id"]: (
                    "ambiguous"
                    if top_score - float(record.get(score_key) or 0.0) <= margin_threshold
                    else "blank"
                )
                for record in records
            }
        else:
            labels = {
                record["spec_id"]: "filled" if index == 0 else "blank"
                for index, record in enumerate(records)
            }

        for record in records:
            new_label = labels[record["spec_id"]]
            if record["prelabel"] == new_label:
                continue

            move_crop_if_saved(record, new_label, output_dir=output_dir, project_root=project_root)
            record["prelabel"] = new_label


def move_crop_if_saved(
    record: dict,
    new_label: str,
    *,
    output_dir: Path | None,
    project_root: Path | None,
) -> None:
    if output_dir is None or project_root is None:
        return
    crop_name = record.get("_crop_name")
    crop_path = record.get("crop_path")
    if not crop_name or not crop_path:
        return

    old_path = project_root / crop_path
    new_path = crop_output_path(output_dir, record["section"], new_label, crop_name)
    new_path.parent.mkdir(parents=True, exist_ok=True)
    if old_path.exists():
        old_path.replace(new_path)
    record["crop_path"] = new_path.relative_to(project_root).as_posix()


_move_crop_if_saved = move_crop_if_saved


def decode_identity(groups: dict[str, list[dict]], field: str, digit_count: int) -> dict:
    columns = []
    status_counts: Counter[str] = Counter()
    value_chars = []
    for index in range(1, digit_count + 1):
        item_id = f"{field}_{index:02d}"
        decoded = decode_group(groups.get(item_id, []))
        decoded["slot"] = index
        columns.append(decoded)
        status_counts[decoded["status"]] += 1
        if decoded.get("selected") is not None:
            value_chars.append(decoded["selected"])
        elif decoded["status"] == "blank":
            value_chars.append("_")
        else:
            value_chars.append("?")

    if all(column["status"] == "accepted" for column in columns):
        status = "accepted"
    elif any(column["status"] in {"need_review", "multi_mark"} for column in columns):
        status = "need_review"
    else:
        status = "incomplete"

    return {
        "field": field,
        "value": "".join(value_chars),
        "status": status,
        "columns": columns,
        "counts": dict(sorted(status_counts.items())),
    }
