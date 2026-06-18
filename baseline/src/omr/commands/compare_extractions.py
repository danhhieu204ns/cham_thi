"""Compare two extraction JSONL files, optionally against ground truth."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from omr.cli_utils import write_json_file, write_markdown_lines
from omr.jsonl_io import read_jsonl_records


IDENTITY_FIELDS = ("sbd", "exam_code")
PART1_IDS = tuple(f"I_{index:03d}" for index in range(1, 41))
PART2_IDS = tuple(
    f"II_{question:03d}_{statement}"
    for question in range(1, 9)
    for statement in ("a", "b", "c", "d")
)
PART3_IDS = tuple(f"III_{index:03d}" for index in range(1, 7))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-a", required=True, help="First extraction JSONL.")
    parser.add_argument("--method-b", required=True, help="Second extraction JSONL.")
    parser.add_argument("--label-a", default="method_a")
    parser.add_argument("--label-b", default="method_b")
    parser.add_argument(
        "--ground-truth",
        default="../web_demo/data/ground_truth.json",
        help="Optional ground-truth JSON. Use empty string to disable.",
    )
    parser.add_argument("--output-dir", default="reports/compare_methods")
    parser.add_argument("--max-differences", type=int, default=200)
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


def normalize_value(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def review_count(record: dict) -> int:
    total = len(record.get("part1", {}).get("review_items", []))
    for section in ("part2", "part3"):
        counts = record.get(section, {}).get("counts", {})
        total += int(counts.get("need_review", 0))
        total += int(counts.get("multi_mark", 0))

    identity = record.get("identity", {})
    for field in IDENTITY_FIELDS:
        status = identity.get(field, {}).get("status")
        if status and status != "accepted":
            total += 1
    return total


def item_key(section: str, item_id: str) -> str:
    return f"{section}:{item_id}"


def flatten_extraction(record: dict) -> dict[str, dict[str, str | bool | None]]:
    items: dict[str, dict[str, str | bool | None]] = {}

    identity = record.get("identity", {})
    for field in IDENTITY_FIELDS:
        value = identity.get(field, {}).get("value")
        status = identity.get(field, {}).get("status")
        items[item_key("identity", field)] = {
            "value": normalize_value(value),
            "status": status,
            "need_review": status not in {None, "accepted"},
        }

    part1_answers = record.get("part1", {}).get("answers", {})
    for qid in PART1_IDS:
        answer = part1_answers.get(qid, {})
        items[item_key("part1", qid)] = {
            "value": normalize_value(answer.get("selected")),
            "status": answer.get("status"),
            "need_review": bool(answer.get("need_review", False)),
        }

    part2_answers = record.get("part2", {}).get("answers", {})
    for qid in PART2_IDS:
        answer = part2_answers.get(qid, {})
        items[item_key("part2", qid)] = {
            "value": normalize_value(answer.get("selected")),
            "status": answer.get("status"),
            "need_review": bool(answer.get("need_review", False)),
        }

    part3_answers = record.get("part3", {}).get("answers", {})
    for qid in PART3_IDS:
        answer = part3_answers.get(qid, {})
        items[item_key("part3", qid)] = {
            "value": normalize_value(answer.get("value")),
            "status": answer.get("status"),
            "need_review": answer.get("status") in {"need_review", "multi_mark"},
        }

    return items


def load_extractions(path: Path) -> dict[str, dict]:
    records = {}
    for record in read_jsonl_records(path):
        image_id = str(record.get("image_id", ""))
        if not image_id:
            continue
        records[image_id] = record
    return records


def load_ground_truth(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}

    raw = json.loads(path.read_text(encoding="utf-8"))
    sheets = raw.get("sheets", raw)
    ground_truth: dict[str, dict[str, str]] = {}
    for image_id, sheet in sheets.items():
        items = {}
        identity = sheet.get("identity", {})
        for field in IDENTITY_FIELDS:
            items[item_key("identity", field)] = normalize_value(identity.get(field))

        answers = sheet.get("answers", {})
        for qid in PART1_IDS:
            items[item_key("part1", qid)] = normalize_value(answers.get(qid))

        part2_answers = sheet.get("part2", {}).get("answers", {})
        for qid in PART2_IDS:
            items[item_key("part2", qid)] = normalize_value(part2_answers.get(qid))

        part3_answers = sheet.get("part3", {}).get("answers", {})
        for qid in PART3_IDS:
            items[item_key("part3", qid)] = normalize_value(part3_answers.get(qid))

        ground_truth[str(image_id)] = items
    return ground_truth


def section_from_key(key: str) -> str:
    return key.split(":", 1)[0]


def empty_metric() -> dict[str, int | float]:
    return {"total": 0, "correct": 0, "incorrect": 0, "accuracy": 0.0}


def finalize_metric(metric: dict[str, int | float]) -> dict[str, int | float]:
    total = int(metric["total"])
    correct = int(metric["correct"])
    metric["incorrect"] = total - correct
    metric["accuracy"] = round(correct / total, 6) if total else 0.0
    return metric


def evaluate_against_ground_truth(
    records: dict[str, dict],
    ground_truth: dict[str, dict[str, str]],
    sheet_ids: Iterable[str] | None = None,
) -> tuple[dict, list[dict]]:
    metrics = {
        "overall": empty_metric(),
        "by_section": {section: empty_metric() for section in ("identity", "part1", "part2", "part3")},
    }
    errors: list[dict] = []
    candidate_ids = set(records) if sheet_ids is None else set(sheet_ids)
    common_ids = sorted(candidate_ids & set(records) & set(ground_truth))

    for image_id in common_ids:
        record = records[image_id]
        extracted = flatten_extraction(record)
        for key, expected in ground_truth[image_id].items():
            section = section_from_key(key)
            actual = normalize_value(extracted.get(key, {}).get("value"))
            correct = actual == expected

            metrics["overall"]["total"] += 1
            metrics["by_section"][section]["total"] += 1
            if correct:
                metrics["overall"]["correct"] += 1
                metrics["by_section"][section]["correct"] += 1
            else:
                errors.append(
                    {
                        "image_id": image_id,
                        "file_name": record.get("file_name"),
                        "section": section,
                        "item_id": key.split(":", 1)[1],
                        "expected": expected,
                        "actual": actual,
                        "status": extracted.get(key, {}).get("status"),
                    }
                )

    finalize_metric(metrics["overall"])
    for metric in metrics["by_section"].values():
        finalize_metric(metric)

    metrics["source_sheet_count"] = len(records)
    metrics["sheet_count"] = len(common_ids)
    metrics["ground_truth_sheet_count"] = len(common_ids)
    metrics["review_item_count"] = sum(review_count(records[image_id]) for image_id in common_ids)
    metrics["status_counts"] = status_counts(records[image_id] for image_id in common_ids)
    return metrics, errors


def status_counts(records: Iterable[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        for item in flatten_extraction(record).values():
            section = ""
            status = item.get("status")
            if status:
                counts[str(status)] += 1
    return dict(sorted(counts.items()))


def compare_pair(
    records_a: dict[str, dict],
    records_b: dict[str, dict],
    ground_truth: dict[str, dict[str, str]],
) -> tuple[dict, list[dict]]:
    common_ids = sorted(set(records_a) & set(records_b))
    summary = {
        "common_sheet_count": len(common_ids),
        "only_a_sheet_count": len(set(records_a) - set(records_b)),
        "only_b_sheet_count": len(set(records_b) - set(records_a)),
        "overall": {"total": 0, "same": 0, "different": 0, "same_rate": 0.0},
        "by_section": {
            section: {"total": 0, "same": 0, "different": 0, "same_rate": 0.0}
            for section in ("identity", "part1", "part2", "part3")
        },
    }
    differences: list[dict] = []

    for image_id in common_ids:
        flat_a = flatten_extraction(records_a[image_id])
        flat_b = flatten_extraction(records_b[image_id])
        keys = sorted(set(flat_a) | set(flat_b))
        gt_items = ground_truth.get(image_id, {})
        for key in keys:
            section = section_from_key(key)
            value_a = normalize_value(flat_a.get(key, {}).get("value"))
            value_b = normalize_value(flat_b.get(key, {}).get("value"))
            same = value_a == value_b

            summary["overall"]["total"] += 1
            summary["by_section"][section]["total"] += 1
            if same:
                summary["overall"]["same"] += 1
                summary["by_section"][section]["same"] += 1
            else:
                summary["overall"]["different"] += 1
                summary["by_section"][section]["different"] += 1
                expected = gt_items.get(key)
                differences.append(
                    {
                        "image_id": image_id,
                        "file_name": records_a[image_id].get("file_name") or records_b[image_id].get("file_name"),
                        "section": section,
                        "item_id": key.split(":", 1)[1],
                        "method_a_value": value_a,
                        "method_a_status": flat_a.get(key, {}).get("status"),
                        "method_b_value": value_b,
                        "method_b_status": flat_b.get(key, {}).get("status"),
                        "expected": expected,
                        "method_a_correct": value_a == expected if expected is not None else None,
                        "method_b_correct": value_b == expected if expected is not None else None,
                    }
                )

    finalize_pair_metric(summary["overall"])
    for metric in summary["by_section"].values():
        finalize_pair_metric(metric)
    return summary, differences


def finalize_pair_metric(metric: dict[str, int | float]) -> None:
    total = int(metric["total"])
    same = int(metric["same"])
    metric["same_rate"] = round(same / total, 6) if total else 0.0


def write_differences_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "file_name",
        "section",
        "item_id",
        "method_a_value",
        "method_a_status",
        "method_b_value",
        "method_b_status",
        "expected",
        "method_a_correct",
        "method_b_correct",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_errors_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "file_name",
        "section",
        "item_id",
        "expected",
        "actual",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_line(label: str, metric: dict) -> str:
    return (
        f"- {label}: {metric['correct']}/{metric['total']} "
        f"({metric['accuracy']:.4f})"
    )


def pair_line(label: str, metric: dict) -> str:
    return (
        f"- {label}: same {metric['same']}/{metric['total']} "
        f"({metric['same_rate']:.4f}), different {metric['different']}"
    )


def build_markdown(result: dict, max_differences: int) -> list[str]:
    label_a = result["labels"]["method_a"]
    label_b = result["labels"]["method_b"]
    pair = result["pairwise"]
    lines = [
        "# Extraction Comparison",
        "",
        f"Generated at: {result['generated_at']}",
        f"Method A: `{label_a}`",
        f"Method B: `{label_b}`",
        "",
        "## Pairwise",
        "",
        pair_line("Overall", pair["overall"]),
    ]
    for section, metric in pair["by_section"].items():
        lines.append(pair_line(section, metric))

    if result.get("ground_truth_enabled"):
        lines.extend(["", "## Ground Truth Accuracy", ""])
        lines.append("Scope: common sheets present in both methods and ground truth.")
        lines.append("")
        for label, metrics in result["methods"].items():
            lines.append(f"### {label}")
            lines.append(metric_line("Overall", metrics["overall"]))
            for section, metric in metrics["by_section"].items():
                lines.append(metric_line(section, metric))
            lines.append(f"- Compared sheets: {metrics['sheet_count']}")
            lines.append(f"- Review items: {metrics['review_item_count']}")
            lines.append("")

    lines.extend(
        [
            "## Differences",
            "",
            f"- Difference CSV: `{result['files']['differences_csv']}`",
            f"- Total differences: {pair['overall']['different']}",
        ]
    )

    for row in result["sample_differences"][:max_differences]:
        expected = row.get("expected")
        suffix = f", gt={expected!r}" if expected is not None else ""
        lines.append(
            "- "
            f"`{row['image_id']}` {row['section']} {row['item_id']}: "
            f"{label_a}={row['method_a_value']!r}, "
            f"{label_b}={row['method_b_value']!r}{suffix}"
        )
    return lines


def main() -> int:
    args = parse_args()
    method_a_path = resolve_project_path(args.method_a)
    method_b_path = resolve_project_path(args.method_b)
    ground_truth_path = resolve_project_path(args.ground_truth) if args.ground_truth else None
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records_a = load_extractions(method_a_path)
    records_b = load_extractions(method_b_path)
    ground_truth = load_ground_truth(ground_truth_path)

    common_eval_ids = sorted(set(records_a) & set(records_b))
    metrics_a, errors_a = (
        evaluate_against_ground_truth(records_a, ground_truth, common_eval_ids)
        if ground_truth
        else ({}, [])
    )
    metrics_b, errors_b = (
        evaluate_against_ground_truth(records_b, ground_truth, common_eval_ids)
        if ground_truth
        else ({}, [])
    )
    pair_summary, differences = compare_pair(records_a, records_b, ground_truth)

    differences_csv = output_dir / "differences.csv"
    write_differences_csv(differences_csv, differences)
    if ground_truth:
        write_errors_csv(output_dir / f"{args.label_a}_errors.csv", errors_a)
        write_errors_csv(output_dir / f"{args.label_b}_errors.csv", errors_b)

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "labels": {"method_a": args.label_a, "method_b": args.label_b},
        "inputs": {
            "method_a": rel(method_a_path),
            "method_b": rel(method_b_path),
            "ground_truth": rel(ground_truth_path) if ground_truth_path is not None else None,
        },
        "ground_truth_enabled": bool(ground_truth),
        "ground_truth_evaluation_scope": "common_method_sheets",
        "methods": {
            args.label_a: metrics_a,
            args.label_b: metrics_b,
        },
        "pairwise": pair_summary,
        "sample_differences": differences[: args.max_differences],
        "files": {
            "summary_json": rel(output_dir / "comparison_summary.json"),
            "summary_md": rel(output_dir / "summary.md"),
            "differences_csv": rel(differences_csv),
            "method_a_errors_csv": rel(output_dir / f"{args.label_a}_errors.csv") if ground_truth else None,
            "method_b_errors_csv": rel(output_dir / f"{args.label_b}_errors.csv") if ground_truth else None,
        },
    }

    write_json_file(result, output_dir / "comparison_summary.json")
    write_markdown_lines(build_markdown(result, args.max_differences), output_dir / "summary.md")

    print(f"summary={output_dir / 'summary.md'}")
    print(f"differences={pair_summary['overall']['different']}")
    if ground_truth:
        print(
            f"{args.label_a}_accuracy={metrics_a['overall']['accuracy']:.6f} "
            f"{args.label_b}_accuracy={metrics_b['overall']['accuracy']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
