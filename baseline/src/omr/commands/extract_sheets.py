"""Batch extraction command for answer-sheet information."""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from omr.cli_utils import (
    filter_ok_records,
    load_selected_ids,
    write_json_file,
    write_markdown_lines,
)
from omr.jsonl_io import read_jsonl_records, write_jsonl_records
from omr.sheet_pipeline import ExtractionThresholds, extract_sheet
from omr.template import load_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/labels/sheets.jsonl")
    parser.add_argument("--template", default="data/labels/template_tnthpt.json")
    parser.add_argument(
        "--image-list",
        default="data/labels/template_samples.txt",
        help="Optional selected-image list.",
    )
    parser.add_argument("--all", action="store_true", help="Process every ok sheet.")
    parser.add_argument(
        "--output-jsonl",
        default="data/processed/results/sheet_extraction_baseline.jsonl",
    )
    parser.add_argument(
        "--warped-output-dir",
        default="data/processed/warped/sheet_extraction",
    )
    parser.add_argument(
        "--crop-output-dir",
        default="",
        help="Optional directory for debug bubble crops.",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--blank-threshold", type=float, default=0.025)
    parser.add_argument("--filled-threshold", type=float, default=0.04)
    parser.add_argument("--identity-filled-threshold", type=float, default=0.06)
    parser.add_argument("--identity-margin-threshold", type=float, default=0.03)
    parser.add_argument("--contact-sheet-width", type=int, default=320)
    parser.add_argument("--visual-limit", type=int, default=20)
    return parser.parse_args()


def make_contact_sheet(
    warped_paths: list[Path],
    visuals_dir: Path,
    target_width: int,
) -> Path | None:
    if not warped_paths:
        return None

    visuals_dir.mkdir(parents=True, exist_ok=True)
    thumbs: list[Image.Image] = []
    labels: list[str] = []
    for path in warped_paths:
        with Image.open(path) as original:
            image = original.convert("RGB")
            ratio = target_width / image.width
            thumb = image.resize(
                (target_width, int(image.height * ratio)),
                Image.Resampling.LANCZOS,
            )
        thumbs.append(thumb)
        labels.append(path.stem)

    label_height = 20
    columns = 2
    rows = (len(thumbs) + columns - 1) // columns
    cell_height = max(thumb.height for thumb in thumbs) + label_height
    sheet = Image.new("RGB", (columns * target_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)

    for index, thumb in enumerate(thumbs):
        x = (index % columns) * target_width
        y = (index // columns) * cell_height
        sheet.paste(thumb, (x, y))
        draw.text((x + 4, y + thumb.height + 2), labels[index][:38], fill=(0, 0, 0))

    output_path = visuals_dir / "warped_contact_sheet.jpg"
    sheet.save(output_path, quality=88)
    return output_path


def copy_visual_images(
    warped_paths: list[Path],
    visuals_dir: Path,
    project_root: Path,
    limit: int,
) -> list[str]:
    if limit <= 0:
        return []

    image_dir = visuals_dir / "warped_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for path in warped_paths[:limit]:
        target = image_dir / path.name
        shutil.copy2(path, target)
        copied.append(target.relative_to(project_root).as_posix())
    return copied


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


def extraction_summary(records: list[dict]) -> dict:
    ok_records = [record for record in records if record.get("status") == "ok"]
    errors = len(records) - len(ok_records)
    part1_counts: Counter[str] = Counter()
    part2_counts: Counter[str] = Counter()
    part3_counts: Counter[str] = Counter()
    identity_counts: Counter[str] = Counter()
    review_total = 0

    for record in ok_records:
        part1_counts.update(record.get("part1", {}).get("counts", {}))
        part2_counts.update(record.get("part2", {}).get("counts", {}))
        part3_counts.update(record.get("part3", {}).get("counts", {}))
        review_total += review_count(record)
        identity = record.get("identity", {})
        for field in ("sbd", "exam_code"):
            status = identity.get(field, {}).get("status")
            if status:
                identity_counts[f"{field}:{status}"] += 1

    return {
        "total": len(records),
        "ok": len(ok_records),
        "error": errors,
        "review_item_count": review_total,
        "identity_status_counts": dict(sorted(identity_counts.items())),
        "part1_status_counts": dict(sorted(part1_counts.items())),
        "part2_status_counts": dict(sorted(part2_counts.items())),
        "part3_status_counts": dict(sorted(part3_counts.items())),
    }


def main() -> int:
    args = parse_args()
    project_root = PROJECT_ROOT
    metadata_path = project_root / args.metadata
    template_path = project_root / args.template
    image_list_path = None if args.all else project_root / args.image_list
    output_jsonl_path = project_root / args.output_jsonl
    warped_output_dir = project_root / args.warped_output_dir if args.warped_output_dir else None
    crop_output_dir = project_root / args.crop_output_dir if args.crop_output_dir else None
    run_dir = project_root / args.run_dir
    visuals_dir = run_dir / "visuals"
    run_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = load_selected_ids(image_list_path)
    metadata_records = filter_ok_records(read_jsonl_records(metadata_path), selected_ids)
    template = load_template(template_path)
    thresholds = ExtractionThresholds(
        blank=args.blank_threshold,
        filled=args.filled_threshold,
        identity_filled=args.identity_filled_threshold,
        identity_margin=args.identity_margin_threshold,
    )

    extraction_records: list[dict] = []
    warped_paths: list[Path] = []
    for metadata in metadata_records:
        image_path = project_root / metadata["relative_path"]
        warped_output_path = (
            warped_output_dir / f"{metadata['image_id']}.jpg"
            if warped_output_dir is not None
            else None
        )

        try:
            extracted = extract_sheet(
                image_path,
                template,
                project_root=project_root,
                thresholds=thresholds,
                crop_output_dir=crop_output_dir,
                warped_output_path=warped_output_path,
            )
        except Exception as exc:  # noqa: BLE001 - keep per-sheet failure reason
            extracted = {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "status": "error",
                "image_id": metadata["image_id"],
                "source_path": metadata["relative_path"],
                "input_path": metadata["relative_path"],
                "error": str(exc),
            }

        extracted["file_name"] = metadata.get("file_name") or Path(metadata["relative_path"]).name
        extracted["split"] = metadata.get("split")
        extraction_records.append(extracted)
        if extracted.get("status") == "ok" and warped_output_path is not None:
            warped_paths.append(warped_output_path)

    write_jsonl_records(extraction_records, output_jsonl_path)

    contact_sheet_path = make_contact_sheet(
        warped_paths,
        visuals_dir=visuals_dir,
        target_width=args.contact_sheet_width,
    )
    visual_images = copy_visual_images(
        warped_paths,
        visuals_dir=visuals_dir,
        project_root=project_root,
        limit=args.visual_limit,
    )

    summary = extraction_summary(extraction_records)
    passed = summary["total"] > 0 and summary["error"] == 0
    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "passed": passed,
        "metadata": metadata_path.relative_to(project_root).as_posix(),
        "template": template_path.relative_to(project_root).as_posix(),
        "output_jsonl": output_jsonl_path.relative_to(project_root).as_posix(),
        "warped_output_dir": (
            warped_output_dir.relative_to(project_root).as_posix()
            if warped_output_dir is not None
            else None
        ),
        "crop_output_dir": (
            crop_output_dir.relative_to(project_root).as_posix()
            if crop_output_dir is not None
            else None
        ),
        "contact_sheet": (
            contact_sheet_path.relative_to(project_root).as_posix()
            if contact_sheet_path is not None
            else None
        ),
        "visual_images": visual_images,
        **summary,
    }
    write_json_file(result, run_dir / "extraction_result.json")

    errors = [record for record in extraction_records if record.get("status") != "ok"]
    lines = [
        "# Sheet Extraction Baseline",
        "",
        f"Generated at: {result['generated_at']}",
        f"Passed: {'yes' if passed else 'no'}",
        "",
        "## Summary",
        "",
        f"- Total sheets: {result['total']}",
        f"- Extracted successfully: {result['ok']}",
        f"- Errors: {result['error']}",
        f"- Review items: {result['review_item_count']}",
        f"- Identity status counts: `{result['identity_status_counts']}`",
        f"- Part I status counts: `{result['part1_status_counts']}`",
        f"- Part II status counts: `{result['part2_status_counts']}`",
        f"- Part III status counts: `{result['part3_status_counts']}`",
        "",
        "## Files",
        "",
        f"- Extraction output: `{result['output_jsonl']}`",
        f"- Warped output dir: `{result['warped_output_dir']}`",
        f"- Crop output dir: `{result['crop_output_dir']}`",
        f"- Contact sheet: `{result['contact_sheet']}`",
        "",
        "## Errors",
        "",
    ]
    if errors:
        lines.extend(
            f"- `{record['image_id']}`: {record.get('error', 'unknown error')}"
            for record in errors
        )
    else:
        lines.append("- No extraction errors.")
    write_markdown_lines(lines, run_dir / "summary.md")

    print(f"passed={passed}")
    print(f"total={result['total']} ok={result['ok']} error={result['error']}")
    print(f"output={output_jsonl_path}")
    print(f"summary={run_dir / 'summary.md'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
