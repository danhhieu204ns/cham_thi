"""Small JSONL helpers for dict-like pipeline records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def read_jsonl_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_jsonl_records_if_exists(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return read_jsonl_records(path)


def write_jsonl_records(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
