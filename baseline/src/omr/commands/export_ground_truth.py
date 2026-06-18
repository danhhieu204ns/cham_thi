"""Export a compact ground-truth file from a verified extraction JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from omr.cli_utils import write_json_file
from omr.jsonl_io import read_jsonl_records


PART1_IDS = tuple(f"I_{index:03d}" for index in range(1, 41))
PART2_IDS = tuple(
    f"II_{question:03d}_{statement}"
    for question in range(1, 9)
    for statement in ("a", "b", "c", "d")
)
PART3_IDS = tuple(f"III_{index:03d}" for index in range(1, 7))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export manually verified extraction results as benchmark ground truth."
    )
    parser.add_argument("--extraction-jsonl", required=True)
    parser.add_argument("--output", default="data/labels/ground_truth_from_classifier.json")
    parser.add_argument(
        "--verified-by",
        default="manual_review",
        help="Short provenance label written into the output metadata.",
    )
    parser.add_argument(
        "--note",
        default="Exported from bubble-classifier extraction after manual verification.",
    )
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def clean_value(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def status_counts(record: dict) -> Counter[str]:
    counts: Counter[str] = Counter()
    for field in ("sbd", "exam_code"):
        status = record.get("identity", {}).get(field, {}).get("status")
        if status:
            counts[f"identity:{field}:{status}"] += 1
    for qid in PART1_IDS:
        status = record.get("part1", {}).get("answers", {}).get(qid, {}).get("status")
        if status:
            counts[f"part1:{status}"] += 1
    for qid in PART2_IDS:
        status = record.get("part2", {}).get("answers", {}).get(qid, {}).get("status")
        if status:
            counts[f"part2:{status}"] += 1
    for qid in PART3_IDS:
        status = record.get("part3", {}).get("answers", {}).get(qid, {}).get("status")
        if status:
            counts[f"part3:{status}"] += 1
    return counts


def sheet_from_record(record: dict) -> dict:
    identity = record.get("identity", {})
    part1_answers = record.get("part1", {}).get("answers", {})
    part2_answers = record.get("part2", {}).get("answers", {})
    part3_answers = record.get("part3", {}).get("answers", {})

    return {
        "file_name": record.get("file_name"),
        "source_path": record.get("source_path"),
        "identity": {
            "sbd": clean_value(identity.get("sbd", {}).get("value")),
            "exam_code": clean_value(identity.get("exam_code", {}).get("value")),
        },
        "answers": {
            qid: clean_value(part1_answers.get(qid, {}).get("selected"))
            for qid in PART1_IDS
        },
        "part2": {
            "answers": {
                qid: clean_value(part2_answers.get(qid, {}).get("selected"))
                for qid in PART2_IDS
            }
        },
        "part3": {
            "answers": {
                qid: clean_value(part3_answers.get(qid, {}).get("value"))
                for qid in PART3_IDS
            }
        },
    }


def main() -> int:
    args = parse_args()
    extraction_path = resolve_project_path(args.extraction_jsonl)
    output_path = resolve_project_path(args.output)

    records = [
        record
        for record in read_jsonl_records(extraction_path)
        if record.get("status") == "ok" and record.get("image_id")
    ]
    sheets = {str(record["image_id"]): sheet_from_record(record) for record in records}

    aggregate_status_counts: Counter[str] = Counter()
    for record in records:
        aggregate_status_counts.update(status_counts(record))

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_extraction_jsonl": rel(extraction_path),
        "verified_by": args.verified_by,
        "note": args.note,
        "sheet_count": len(sheets),
        "status_counts": dict(sorted(aggregate_status_counts.items())),
        "sheets": dict(sorted(sheets.items())),
    }
    write_json_file(payload, output_path, sort_keys=True)

    print(f"output={output_path}")
    print(f"sheets={len(sheets)}")
    print(f"status_counts={json.dumps(payload['status_counts'], ensure_ascii=False, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
