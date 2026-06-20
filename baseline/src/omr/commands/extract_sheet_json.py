"""Run the full single-sheet extraction pipeline and write JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from omr.cli_utils import write_json_file
from omr.sheet_pipeline import BubbleModelSettings, ExtractionThresholds, extract_sheet
from omr.template import load_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to one raw answer-sheet image.")
    parser.add_argument("--template", default="data/labels/template_tnthpt.json")
    parser.add_argument("--output", default="", help="Optional output JSON path. Defaults to stdout.")
    parser.add_argument("--warped-output", default="", help="Optional path to save the warped sheet image.")
    parser.add_argument("--bbox-output", default="", help="Optional path to save warped sheet with bubble bboxes.")
    parser.add_argument("--crop-output-dir", default="", help="Optional directory to save all bubble crops.")
    parser.add_argument(
        "--debug-output-dir",
        default="",
        help="Optional directory for per-image step-by-step debug images.",
    )
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
    return parser.parse_args()


def resolve_input_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (PROJECT_ROOT / path).resolve()


def resolve_project_path(value: str) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


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


def emit_result(result: dict, output_path: Path | None) -> None:
    if output_path is None:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    write_json_file(result, output_path)
    print(f"output={output_path}")


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


def main() -> int:
    args = parse_args()
    template = load_template(resolve_project_path(args.template) or PROJECT_ROOT / args.template)
    output_path = resolve_project_path(args.output)
    warped_output_path = resolve_project_path(args.warped_output)
    bbox_output_path = resolve_project_path(args.bbox_output)
    crop_output_dir = resolve_project_path(args.crop_output_dir)
    debug_output_dir = resolve_project_path(args.debug_output_dir)
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

    try:
        result = extract_sheet(
            resolve_input_path(args.input_path),
            template,
            project_root=PROJECT_ROOT,
            thresholds=thresholds,
            bubble_classifier=bubble_classifier,
            bubble_model_settings=bubble_model_settings,
            crop_output_dir=crop_output_dir,
            warped_output_path=warped_output_path,
            bbox_overlay_path=bbox_output_path,
            layout_detector=layout_detector,
            debug_output_dir=debug_output_dir,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should return machine-readable failure
        result = {
            "status": "error",
            "input_path": args.input_path,
            "error": str(exc),
        }
        emit_result(result, output_path)
        return 1

    emit_result(result, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
