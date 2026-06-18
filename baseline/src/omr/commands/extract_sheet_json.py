"""Run the full single-sheet extraction pipeline and write JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from omr.cli_utils import write_json_file
from omr.sheet_pipeline import ExtractionThresholds, extract_sheet
from omr.template import load_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to one raw answer-sheet image.")
    parser.add_argument("--template", default="data/labels/template_tnthpt.json")
    parser.add_argument("--output", default="", help="Optional output JSON path. Defaults to stdout.")
    parser.add_argument("--warped-output", default="", help="Optional path to save the warped sheet image.")
    parser.add_argument("--crop-output-dir", default="", help="Optional directory to save all bubble crops.")
    parser.add_argument("--blank-threshold", type=float, default=0.025)
    parser.add_argument("--filled-threshold", type=float, default=0.04)
    parser.add_argument("--identity-filled-threshold", type=float, default=0.06)
    parser.add_argument("--identity-margin-threshold", type=float, default=0.03)
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


def emit_result(result: dict, output_path: Path | None) -> None:
    if output_path is None:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    write_json_file(result, output_path)
    print(f"output={output_path}")


def main() -> int:
    args = parse_args()
    template = load_template(resolve_project_path(args.template) or PROJECT_ROOT / args.template)
    output_path = resolve_project_path(args.output)
    warped_output_path = resolve_project_path(args.warped_output)
    crop_output_dir = resolve_project_path(args.crop_output_dir)
    thresholds = ExtractionThresholds(
        blank=args.blank_threshold,
        filled=args.filled_threshold,
        identity_filled=args.identity_filled_threshold,
        identity_margin=args.identity_margin_threshold,
    )

    try:
        result = extract_sheet(
            resolve_input_path(args.input_path),
            template,
            project_root=PROJECT_ROOT,
            thresholds=thresholds,
            crop_output_dir=crop_output_dir,
            warped_output_path=warped_output_path,
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
