from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = "bubble_classifier/data"
DEFAULT_OUTPUT = "bubble_classifier/data/manifest.csv"
DEFAULT_STATS = "bubble_classifier/data/manifest_stats.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABELS = ("blank", "filled")

FIELDNAMES = [
    "crop_path",
    "gt_label",
    "prelabel",
    "darkness_score",
    "split",
    "image_id",
    "file_name",
    "section",
    "field",
    "group_id",
    "choice",
    "group_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a bubble classifier manifest from class folders."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--stats-output", default=DEFAULT_STATS)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def relative_repo_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def split_for_image(image_id: str, *, val_ratio: float, test_ratio: float) -> str:
    digest = hashlib.sha1(image_id.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    if value < test_ratio:
        return "test"
    if value < test_ratio + val_ratio:
        return "val"
    return "train"


def parse_crop_metadata(path: Path, label_dir: Path) -> dict[str, str]:
    rel = path.relative_to(label_dir)
    section = rel.parts[0] if len(rel.parts) > 1 else ""
    stem = path.stem
    image_id, _, tail = stem.partition("__")
    if not image_id:
        image_id = stem

    field = ""
    group_id = ""
    choice = ""
    if tail:
        parts = tail.split("_")
        choice = parts[-1] if parts else ""
        group_id = "_".join(parts[:-1])
        if section == "identity" and len(parts) >= 2:
            field = "_".join(parts[:-2])
        elif section == "part1":
            field = "answer"
        elif section == "part2":
            field = "answer"
        elif section == "part3" and len(parts) >= 3:
            field = parts[2]

    return {
        "image_id": image_id,
        "file_name": path.name,
        "section": section,
        "field": field,
        "group_id": group_id,
        "choice": choice,
    }


def iter_rows(data_dir: Path, *, val_ratio: float, test_ratio: float) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label in LABELS:
        label_dir = data_dir / label
        if not label_dir.is_dir():
            raise SystemExit(f"missing label directory: {label_dir}")

        for path in sorted(label_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            metadata = parse_crop_metadata(path, label_dir)
            split = split_for_image(
                metadata["image_id"],
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )
            rows.append(
                {
                    "crop_path": relative_repo_path(path),
                    "gt_label": label,
                    "prelabel": "",
                    "darkness_score": "",
                    "split": split,
                    "group_status": "folder_label",
                    **metadata,
                }
            )
    rows.sort(key=lambda row: (row["split"], row["image_id"], row["gt_label"], row["crop_path"]))
    return rows


def write_manifest(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_stats(rows: list[dict[str, str]], stats_path: Path, data_dir: Path) -> dict:
    image_ids = {row["image_id"] for row in rows}
    stats = {
        "source": str(data_dir.relative_to(REPO_ROOT)),
        "image_id_count": len(image_ids),
        "crop_count": len(rows),
        "by_label": dict(sorted(Counter(row["gt_label"] for row in rows).items())),
        "by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
        "by_section": dict(sorted(Counter(row["section"] for row in rows).items())),
        "by_split_label": dict(
            sorted(Counter(f"{row['split']}:{row['gt_label']}" for row in rows).items())
        ),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> int:
    args = parse_args()
    data_dir = repo_path(args.data_dir)
    output_path = repo_path(args.output)
    stats_path = repo_path(args.stats_output)

    rows = iter_rows(data_dir, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    write_manifest(rows, output_path)
    stats = write_stats(rows, stats_path, data_dir)

    print(f"manifest={output_path}")
    print(f"stats={stats_path}")
    print(f"crops={len(rows)}")
    print(f"image_ids={stats['image_id_count']}")
    print(f"labels={stats['by_label']}")
    print(f"splits={stats['by_split']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
