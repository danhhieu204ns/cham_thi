"""Shared helpers for small pipeline command-line scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_selected_ids(path: Path | None) -> set[str] | None:
    """Read the first tab-separated field from a selected-image list."""
    if path is None or not path.is_file():
        return None

    selected: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        selected.add(line.split("\t")[0])
    return selected


def filter_ok_records(
    records: Iterable[dict],
    selected_ids: set[str] | None = None,
    *,
    sort_field: str = "relative_path",
) -> list[dict]:
    filtered = [
        record
        for record in records
        if record.get("status") == "ok"
        and (selected_ids is None or record["image_id"] in selected_ids)
    ]
    filtered.sort(key=lambda item: str(item.get(sort_field, item["image_id"])).lower())
    return filtered


def write_json_file(data: dict, path: Path, *, sort_keys: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=sort_keys) + "\n",
        encoding="utf-8",
    )


def write_markdown_lines(lines: Iterable[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
