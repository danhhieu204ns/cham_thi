"""Build compact extraction data for the static web demo."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from omr.cli_utils import write_json_file
from omr.jsonl_io import read_jsonl_records
from omr.sheet_pipeline import build_all_specs


QUESTION_COUNT = 40
CHOICES = ("A", "B", "C", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/labels/sheets.jsonl")
    parser.add_argument(
        "--extraction-jsonl",
        default="data/processed/results/sheet_extraction_baseline.jsonl",
    )
    parser.add_argument("--template", default="data/labels/template_tnthpt.json")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--output", default="../web_demo/data/demo_data.json")
    return parser.parse_args()


def rel(path: str | None) -> str | None:
    return path.replace("\\", "/") if path else path


def section_order(section: str) -> int:
    return {
        "identity": 0,
        "part1": 1,
        "part2": 2,
        "part3": 3,
    }.get(section, 99)


def build_bubble_specs(template: dict) -> list[dict]:
    specs = []
    for record in build_all_specs(template):
        specs.append(
            {
                "section": record.get("section"),
                "field": record.get("field"),
                "spec_id": record.get("spec_id"),
                "item_id": record.get("item_id"),
                "question_id": record.get("question_id"),
                "question_number": record.get("question_number"),
                "choice": record.get("choice"),
                "label": record.get("label", record.get("choice")),
                "role": record.get("role", "answer"),
                "slot": record.get("slot"),
                "statement": record.get("statement"),
                "center": record.get("center"),
                "bbox": record.get("bbox"),
            }
        )
    return sorted(
        specs,
        key=lambda item: (
            section_order(str(item["section"])),
            item.get("field") or "",
            item.get("question_number") or 0,
            item.get("slot") or 0,
            str(item.get("choice") or ""),
        ),
    )


def compact_part1_answer(qid: str, decoded_answer: dict | None) -> dict:
    decoded_answer = decoded_answer or {}
    states = {}
    for choice in CHOICES:
        state = decoded_answer.get("states", {}).get(choice, {})
        states[choice] = {
            "prelabel": state.get("prelabel"),
            "darkness_score": state.get("darkness_score"),
            "crop_path": rel(state.get("crop_path")),
        }

    return {
        "question_id": qid,
        "question_number": decoded_answer.get("question_number") or int(qid.split("_")[1]),
        "selected": decoded_answer.get("selected"),
        "status": decoded_answer.get("status"),
        "need_review": decoded_answer.get("need_review", False),
        "review_reasons": decoded_answer.get("review_reasons", []),
        "darkness_score": decoded_answer.get("darkness_score"),
        "darkness_margin": decoded_answer.get("darkness_margin"),
        "states": states,
    }


def review_count(record: dict) -> int:
    total = len(record.get("part1", {}).get("review_items", []))
    for section in ("part2", "part3"):
        counts = record.get(section, {}).get("counts", {})
        total += int(counts.get("need_review", 0))
        total += int(counts.get("multi_mark", 0))

    identity = record.get("identity", {})
    for field in ("sbd", "exam_code"):
        status = identity.get(field, {}).get("status")
        if status and status != "accepted":
            total += 1
    return total


def load_extraction_summary(project_root: Path, run_dir_arg: str) -> dict:
    if not run_dir_arg:
        return {}
    path = project_root / run_dir_arg / "extraction_result.json"
    if not path.is_file():
        return {}
    result = json.loads(path.read_text(encoding="utf-8"))
    result["run_dir"] = (project_root / run_dir_arg).relative_to(project_root).as_posix()
    return result


def main() -> int:
    args = parse_args()
    project_root = PROJECT_ROOT

    metadata = {item["image_id"]: item for item in read_jsonl_records(project_root / args.metadata)}
    extraction_records = [
        item
        for item in read_jsonl_records(project_root / args.extraction_jsonl)
        if item.get("status") == "ok"
    ]
    template = json.loads((project_root / args.template).read_text(encoding="utf-8"))
    summary = load_extraction_summary(project_root, args.run_dir)

    sheets: list[dict] = []
    aggregate_part1_counts: Counter[str] = Counter()
    aggregate_part2_counts: Counter[str] = Counter()
    aggregate_part3_counts: Counter[str] = Counter()
    aggregate_identity_counts: Counter[str] = Counter()
    review_total = 0

    for record in sorted(extraction_records, key=lambda item: item.get("image_id", "")):
        image_id = record["image_id"]
        meta_record = metadata.get(image_id, {})
        part1 = record.get("part1", {})
        answers = {
            f"I_{question_number:03d}": compact_part1_answer(
                f"I_{question_number:03d}",
                part1.get("answers", {}).get(f"I_{question_number:03d}"),
            )
            for question_number in range(1, QUESTION_COUNT + 1)
        }

        aggregate_part1_counts.update(part1.get("counts", {}))
        aggregate_part2_counts.update(record.get("part2", {}).get("counts", {}))
        aggregate_part3_counts.update(record.get("part3", {}).get("counts", {}))
        for field in ("sbd", "exam_code"):
            status = record.get("identity", {}).get(field, {}).get("status")
            if status:
                aggregate_identity_counts[f"{field}:{status}"] += 1
        review_total += review_count(record)

        sheets.append(
            {
                "image_id": image_id,
                "file_name": record.get("file_name")
                or meta_record.get("file_name")
                or Path(record["source_path"]).name,
                "split": record.get("split") or meta_record.get("split"),
                "source_path": rel(record.get("source_path") or meta_record.get("relative_path")),
                "warped_path": rel(record.get("warp", {}).get("warped_path") or record.get("input_path")),
                "counts": part1.get("counts", {}),
                "review_items": part1.get("review_items", []),
                "answers": answers,
                "identity": record.get("identity", {}),
                "part2": record.get("part2", {"answers": {}, "counts": {}}),
                "part3": record.get("part3", {"answers": {}, "counts": {}}),
            }
        )

    generated = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_files": {
            "metadata": rel(args.metadata),
            "extraction_jsonl": rel(args.extraction_jsonl),
            "template": rel(args.template),
        },
        "summary": {
            "sheet_count": len(sheets),
            "review_item_count": review_total,
            "identity_status_counts": dict(sorted(aggregate_identity_counts.items())),
            "part1_status_counts": dict(sorted(aggregate_part1_counts.items())),
            "part2_status_counts": dict(sorted(aggregate_part2_counts.items())),
            "part3_status_counts": dict(sorted(aggregate_part3_counts.items())),
            **summary,
        },
        "template": {
            "template_id": template.get("template_id"),
            "canonical_size": template["coordinate_system"]["canonical_size"],
            "regions": template["regions"],
        },
        "bubble_specs": build_bubble_specs(template),
        "sheets": sheets,
    }

    output_path = project_root / args.output
    write_json_file(generated, output_path, sort_keys=True)
    print(f"output={output_path}")
    print(f"sheets={len(sheets)}")
    print(f"bubble_specs={len(generated['bubble_specs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
