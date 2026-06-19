"""Relabel bubble groups by a ranked confidence score."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .identity import move_crop_if_saved


def group_by_item(records: Iterable[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["item_id"]].append(record)
    return groups


def relabel_groups_by_score(
    crop_records: list[dict],
    *,
    filled_threshold: float,
    margin_threshold: float,
    score_key: str,
    output_dir: Path | None = None,
    project_root: Path | None = None,
) -> None:
    groups = group_by_item(crop_records)
    for records in groups.values():
        records.sort(key=lambda record: float(record.get(score_key) or 0.0), reverse=True)
        if not records:
            continue

        top_score = float(records[0].get(score_key) or 0.0)
        second_score = float(records[1].get(score_key) or 0.0) if len(records) > 1 else 0.0
        margin = top_score - second_score
        if top_score < filled_threshold:
            labels = {record["spec_id"]: "blank" for record in records}
        elif len(records) > 1 and margin < margin_threshold:
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
            record["group_score_key"] = score_key
            record["group_top_score"] = round(top_score, 6)
            record["group_second_score"] = round(second_score, 6)
            record["group_score_margin"] = round(margin, 6)
            record["group_filled_threshold"] = filled_threshold
            record["group_margin_threshold"] = margin_threshold
            record["group_relabel_source"] = "ranked_score"
            new_label = labels[record["spec_id"]]
            if record["prelabel"] != new_label:
                move_crop_if_saved(record, new_label, output_dir=output_dir, project_root=project_root)
                record["prelabel"] = new_label
            if record.get("prelabel_source") == "adaptive_rule":
                record["prelabel_source"] = "adaptive_rule_group"
