from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bubble_classifier.inference import (
    DEFAULT_FILLED_THRESHOLD,
    DEFAULT_MODEL_PATH,
    BubbleClassifier,
    iter_image_paths,
    repo_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict bubble states with the trained ResNet18 model.")
    parser.add_argument("inputs", nargs="*", help="Image files or directories to classify.")
    parser.add_argument("--manifest", help="CSV containing a crop_path column to classify.")
    parser.add_argument("--crop-column", default="crop_path")
    parser.add_argument("--output", help="Output CSV path. Defaults to stdout.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--threshold", type=float, default=DEFAULT_FILLED_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(rows: list[dict[str, object]], output: str | None) -> None:
    if not rows:
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    if output:
        output_path = repo_path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"predictions={output_path}")
        return

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def rows_from_manifest(args: argparse.Namespace, classifier: BubbleClassifier) -> list[dict[str, object]]:
    manifest_path = repo_path(args.manifest)
    manifest_rows = load_manifest_rows(manifest_path)
    if manifest_rows and args.crop_column not in manifest_rows[0]:
        raise SystemExit(f"manifest is missing column: {args.crop_column}")

    image_paths = [row[args.crop_column] for row in manifest_rows]
    predictions = classifier.predict_paths(image_paths, batch_size=args.batch_size)

    output_rows = []
    for source_row, prediction in zip(manifest_rows, predictions):
        output_rows.append({**source_row, **prediction.to_row()})
    return output_rows


def rows_from_inputs(args: argparse.Namespace, classifier: BubbleClassifier) -> list[dict[str, object]]:
    image_paths = iter_image_paths(args.inputs, recursive=args.recursive)
    predictions = classifier.predict_paths(image_paths, batch_size=args.batch_size)
    return [prediction.to_row() for prediction in predictions]


def main() -> int:
    args = parse_args()
    if not args.manifest and not args.inputs:
        raise SystemExit("provide --manifest or at least one image/directory input")

    classifier = BubbleClassifier(args.model, filled_threshold=args.threshold)
    rows = rows_from_manifest(args, classifier) if args.manifest else rows_from_inputs(args, classifier)
    write_rows(rows, args.output)
    print(
        f"count={len(rows)} model={classifier.model_path} threshold={classifier.filled_threshold} "
        f"device={classifier.device}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
