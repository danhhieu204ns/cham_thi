"""Batch extraction command for answer-sheet information."""

from __future__ import annotations

import argparse
import shutil
import sys
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
from omr.sheet_pipeline import BubbleModelSettings, ExtractionThresholds, extract_sheet
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
    parser.add_argument(
        "--bbox-output-dir",
        default="",
        help="Directory for warped sheet bbox overlays. Defaults to run-dir/visuals/bbox_images.",
    )
    parser.add_argument(
        "--debug-output-dir",
        default="",
        help="Optional directory for per-image step-by-step debug images.",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--blank-threshold", type=float, default=0.025)
    parser.add_argument("--filled-threshold", type=float, default=0.05)
    parser.add_argument("--answer-margin-threshold", type=float, default=0.025)
    parser.add_argument("--identity-filled-threshold", type=float, default=0.06)
    parser.add_argument("--identity-margin-threshold", type=float, default=0.03)
    parser.add_argument(
        "--bubble-classifier",
        action="store_true",
        help="Use bubble_classifier ResNet inference instead of rule labels.",
    )
    parser.add_argument(
        "--bubble-model-path",
        default="",
        help="Optional classifier checkpoint path. Defaults to bubble_classifier's best_model.pt.",
    )
    parser.add_argument("--bubble-filled-threshold", type=float, default=0.90)
    parser.add_argument("--bubble-margin-threshold", type=float, default=0.10)
    parser.add_argument("--bubble-batch-size", type=int, default=256)
    parser.add_argument(
        "--layout-checkpoint",
        default="",
        help="Optional layout_v0 checkpoint to use for marker-based warping.",
    )
    parser.add_argument("--layout-device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--layout-marker-threshold", type=float, default=0.25)
    parser.add_argument("--layout-bubble-threshold", type=float, default=0.25)
    parser.add_argument("--layout-max-marker-peaks", type=int, default=40)
    parser.add_argument("--layout-marker-match-tolerance", type=float, default=55.0)
    parser.add_argument("--contact-sheet-width", type=int, default=320)
    parser.add_argument("--visual-limit", type=int, default=20)
    return parser.parse_args()


def load_bubble_classifier(model_path: str, filled_threshold: float):
    repo_root = PROJECT_ROOT.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from bubble_classifier import BubbleClassifier

    kwargs = {"filled_threshold": filled_threshold}
    if model_path:
        path = Path(model_path)
        kwargs["model_path"] = path if path.is_absolute() else repo_root / path
    return BubbleClassifier(**kwargs)


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    project_path = (PROJECT_ROOT / path).resolve()
    if project_path.exists():
        return project_path
    return (PROJECT_ROOT.parent / path).resolve()


def load_layout_detector(args: argparse.Namespace):
    if not args.layout_checkpoint:
        return None

    from omr.layout_inference import LayoutDetector, LayoutDetectorSettings

    settings = LayoutDetectorSettings(
        marker_threshold=args.layout_marker_threshold,
        bubble_threshold=args.layout_bubble_threshold,
        max_marker_peaks=args.layout_max_marker_peaks,
        marker_match_tolerance_px=args.layout_marker_match_tolerance,
    )
    return LayoutDetector(
        resolve_repo_path(args.layout_checkpoint),
        device=args.layout_device,
        settings=settings,
    )


def make_contact_sheet(
    warped_paths: list[Path],
    visuals_dir: Path,
    target_width: int,
    output_name: str,
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

    output_path = visuals_dir / output_name
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
        copied.append(relative_path(target, project_root))
    return copied


def relative_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def relative_paths(paths: list[Path], project_root: Path) -> list[str]:
    values = []
    for path in paths:
        values.append(relative_path(path, project_root))
    return values


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
    auto_pass_total = 0
    confidence_values = []

    for record in ok_records:
        part1_counts.update(record.get("part1", {}).get("counts", {}))
        part2_counts.update(record.get("part2", {}).get("counts", {}))
        part3_counts.update(record.get("part3", {}).get("counts", {}))
        review_total += review_count(record)
        confidence = record.get("confidence", {})
        auto_pass_total += int(bool(confidence.get("auto_pass")))
        if confidence.get("sheet_confidence") is not None:
            confidence_values.append(float(confidence["sheet_confidence"]))
        identity = record.get("identity", {})
        for field in ("sbd", "exam_code"):
            status = identity.get(field, {}).get("status")
            if status:
                identity_counts[f"{field}:{status}"] += 1

    return {
        "total": len(records),
        "ok": len(ok_records),
        "error": errors,
        "auto_pass": auto_pass_total,
        "review_item_count": review_total,
        "min_sheet_confidence": round(min(confidence_values), 6) if confidence_values else None,
        "avg_sheet_confidence": (
            round(sum(confidence_values) / len(confidence_values), 6)
            if confidence_values
            else None
        ),
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
    debug_output_dir = project_root / args.debug_output_dir if args.debug_output_dir else None
    run_dir = project_root / args.run_dir
    visuals_dir = run_dir / "visuals"
    bbox_output_dir = (
        Path(args.bbox_output_dir)
        if args.bbox_output_dir
        else visuals_dir / "bbox_images"
    )
    if not bbox_output_dir.is_absolute():
        bbox_output_dir = project_root / bbox_output_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = load_selected_ids(image_list_path)
    metadata_records = filter_ok_records(read_jsonl_records(metadata_path), selected_ids)
    template = load_template(template_path)
    thresholds = ExtractionThresholds(
        blank=args.blank_threshold,
        filled=args.filled_threshold,
        answer_margin=args.answer_margin_threshold,
        identity_filled=args.identity_filled_threshold,
        identity_margin=args.identity_margin_threshold,
    )
    bubble_classifier = (
        load_bubble_classifier(args.bubble_model_path, args.bubble_filled_threshold)
        if args.bubble_classifier
        else None
    )
    bubble_model_settings = BubbleModelSettings(
        filled_threshold=args.bubble_filled_threshold,
        margin_threshold=args.bubble_margin_threshold,
        batch_size=args.bubble_batch_size,
    )
    layout_detector = load_layout_detector(args)

    extraction_records: list[dict] = []
    warped_paths: list[Path] = []
    bbox_paths: list[Path] = []
    for index, metadata in enumerate(metadata_records, start=1):
        image_path = project_root / metadata["relative_path"]
        warped_output_path = (
            warped_output_dir / f"{metadata['image_id']}.jpg"
            if warped_output_dir is not None
            else None
        )
        bbox_overlay_path = (
            bbox_output_dir / f"{metadata['image_id']}_bbox.jpg"
            if args.visual_limit < 0 or len(bbox_paths) < args.visual_limit
            else None
        )

        try:
            extracted = extract_sheet(
                image_path,
                template,
                project_root=project_root,
                thresholds=thresholds,
                bubble_classifier=bubble_classifier,
                bubble_model_settings=bubble_model_settings,
                crop_output_dir=crop_output_dir,
                warped_output_path=warped_output_path,
                bbox_overlay_path=bbox_overlay_path,
                layout_detector=layout_detector,
                debug_output_dir=debug_output_dir,
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
        if extracted.get("status") == "ok" and bbox_overlay_path is not None:
            bbox_paths.append(bbox_overlay_path)
        print(
            f"{index}/{len(metadata_records)} {metadata['image_id']} status={extracted.get('status')}",
            flush=True,
        )

    write_jsonl_records(extraction_records, output_jsonl_path)

    contact_sheet_path = make_contact_sheet(
        warped_paths,
        visuals_dir=visuals_dir,
        target_width=args.contact_sheet_width,
        output_name="warped_contact_sheet.jpg",
    )
    bbox_contact_sheet_path = make_contact_sheet(
        bbox_paths,
        visuals_dir=visuals_dir,
        target_width=args.contact_sheet_width,
        output_name="bbox_contact_sheet.jpg",
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
        "output_jsonl": relative_path(output_jsonl_path, project_root),
        "bubble_classifier": (
            {
                "model_path": str(getattr(bubble_classifier, "model_path", "unknown")),
                "filled_threshold": args.bubble_filled_threshold,
                "margin_threshold": args.bubble_margin_threshold,
                "batch_size": args.bubble_batch_size,
            }
            if bubble_classifier is not None
            else None
        ),
        "layout_detector": (
            {
                "checkpoint": str(getattr(layout_detector, "checkpoint_path", "unknown")),
                "device": str(getattr(layout_detector, "device", "unknown")),
                "image_size": list(getattr(layout_detector, "image_size", [])),
                "marker_threshold": args.layout_marker_threshold,
                "bubble_threshold": args.layout_bubble_threshold,
                "max_marker_peaks": args.layout_max_marker_peaks,
                "marker_match_tolerance": args.layout_marker_match_tolerance,
            }
            if layout_detector is not None
            else None
        ),
        "warped_output_dir": (
            relative_path(warped_output_dir, project_root)
            if warped_output_dir is not None
            else None
        ),
        "crop_output_dir": (
            relative_path(crop_output_dir, project_root)
            if crop_output_dir is not None
            else None
        ),
        "debug_output_dir": (
            relative_path(debug_output_dir, project_root)
            if debug_output_dir is not None
            else None
        ),
        "contact_sheet": (
            relative_path(contact_sheet_path, project_root)
            if contact_sheet_path is not None
            else None
        ),
        "bbox_contact_sheet": (
            relative_path(bbox_contact_sheet_path, project_root)
            if bbox_contact_sheet_path is not None
            else None
        ),
        "bbox_images": relative_paths(bbox_paths, project_root),
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
        f"- Auto-pass sheets: {result['auto_pass']}",
        f"- Review items: {result['review_item_count']}",
        f"- Sheet confidence: min `{result['min_sheet_confidence']}`, avg `{result['avg_sheet_confidence']}`",
        f"- Identity status counts: `{result['identity_status_counts']}`",
        f"- Part I status counts: `{result['part1_status_counts']}`",
        f"- Part II status counts: `{result['part2_status_counts']}`",
        f"- Part III status counts: `{result['part3_status_counts']}`",
        "",
        "## Files",
        "",
        f"- Extraction output: `{result['output_jsonl']}`",
        f"- Bubble classifier: `{result['bubble_classifier']}`",
        f"- Layout detector: `{result['layout_detector']}`",
        f"- Warped output dir: `{result['warped_output_dir']}`",
        f"- Crop output dir: `{result['crop_output_dir']}`",
        f"- Debug output dir: `{result['debug_output_dir']}`",
        f"- Contact sheet: `{result['contact_sheet']}`",
        f"- Bbox contact sheet: `{result['bbox_contact_sheet']}`",
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
