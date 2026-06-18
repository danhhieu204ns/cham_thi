"""Build a side-by-side HTML visualization for a comparison run."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from omr.jsonl_io import read_jsonl_records
from omr.sheet_pipeline import build_all_specs
from omr.template import load_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "comparison_dir",
        help="Directory containing comparison_summary.json and differences.csv.",
    )
    parser.add_argument("--template", default="data/labels/template_tnthpt.json")
    parser.add_argument("--output", default="", help="Defaults to comparison_dir/side_by_side.html.")
    parser.add_argument("--max-sheets", type=int, default=0, help="0 means all sheets with differences.")
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def rel_from_output(path_value: str | None, output_path: Path) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return Path(os.path.relpath(path.resolve(), output_path.parent.resolve())).as_posix()


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_records(path: Path) -> dict[str, dict]:
    return {
        str(record["image_id"]): record
        for record in read_jsonl_records(path)
        if record.get("image_id")
    }


def build_spec_index(template: dict) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    specs_by_item: dict[str, list[dict]] = defaultdict(list)
    specs_by_choice: dict[str, dict] = {}
    for spec in build_all_specs(template):
        section = spec["section"]
        item_id = spec["item_id"]
        question_id = spec.get("question_id")
        choice = str(spec.get("choice"))

        specs_by_item[f"{section}:{item_id}"].append(spec)
        if section == "part3" and question_id:
            specs_by_item[f"{section}:{question_id}"].append(spec)
        specs_by_choice[f"{section}:{item_id}:{choice}"] = spec
        if section == "part3" and question_id:
            specs_by_choice[f"{section}:{question_id}:{choice}"] = spec
    return specs_by_item, specs_by_choice


def field_region(template: dict, item_id: str) -> list[int] | None:
    if item_id in {"sbd", "exam_code"}:
        region = template.get("regions", {}).get(item_id, {})
        bbox = region.get("bbox")
        if bbox:
            return [int(value) for value in bbox]
    return None


def value_choices(section: str, item_id: str, value: str) -> list[str]:
    if not value:
        return []
    if section in {"part1", "part2"}:
        return [value]
    if section == "part3":
        choices = []
        for index, char in enumerate(value):
            if char == "-":
                choices.append("-")
            elif char == ",":
                choices.append(f"after_{max(1, index)}")
            elif char.isdigit():
                choices.append(char)
        return choices
    return []


def selected_specs_for_method(
    row: dict,
    method_prefix: str,
    specs_by_choice: dict[str, dict],
) -> list[dict]:
    section = row["section"]
    item_id = row["item_id"]
    value = row.get(f"{method_prefix}_value", "")
    specs = []
    for choice in value_choices(section, item_id, value):
        spec = specs_by_choice.get(f"{section}:{item_id}:{choice}")
        if spec is not None:
            specs.append(spec)
    return specs


def overlay_svg(
    row_group: list[dict],
    method_prefix: str,
    template: dict,
    specs_by_item: dict[str, list[dict]],
    specs_by_choice: dict[str, dict],
) -> str:
    size = template["coordinate_system"]["canonical_size"]
    width = int(size["width"])
    height = int(size["height"])
    parts = [
        f'<svg class="overlay" viewBox="0 0 {width} {height}" preserveAspectRatio="none">'
    ]
    for row in row_group:
        section = row["section"]
        item_id = row["item_id"]
        key = f"{section}:{item_id}"
        item_specs = specs_by_item.get(key, [])
        selected_specs = selected_specs_for_method(row, method_prefix, specs_by_choice)

        if section == "identity":
            bbox = field_region(template, item_id)
            if bbox:
                x1, y1, x2, y2 = bbox
                parts.append(rect(x1, y1, x2, y2, "group"))
        else:
            for spec in item_specs:
                x1, y1, x2, y2 = spec["bbox"]
                parts.append(rect(x1, y1, x2, y2, "group"))

        for spec in selected_specs:
            x1, y1, x2, y2 = spec["bbox"]
            parts.append(rect(x1, y1, x2, y2, "selected"))
            label = html.escape(str(spec.get("label") or spec.get("choice") or ""))
            parts.append(
                f'<text x="{x1}" y="{max(12, y1 - 4)}" class="bubble-label">{label}</text>'
            )
    parts.append("</svg>")
    return "\n".join(parts)


def rect(x1: int, y1: int, x2: int, y2: int, kind: str) -> str:
    return (
        f'<rect class="{kind}" x="{x1}" y="{y1}" '
        f'width="{x2 - x1}" height="{y2 - y1}" />'
    )


def sheet_card(
    image_id: str,
    rows: list[dict],
    record_a: dict,
    record_b: dict,
    labels: dict,
    output_path: Path,
    template: dict,
    specs_by_item: dict[str, list[dict]],
    specs_by_choice: dict[str, dict],
) -> str:
    image_a = html.escape(rel_from_output(record_a.get("warp", {}).get("warped_path"), output_path))
    image_b = html.escape(rel_from_output(record_b.get("warp", {}).get("warped_path"), output_path))
    label_a = html.escape(labels["method_a"])
    label_b = html.escape(labels["method_b"])
    table_rows = []
    for row in rows:
        a_correct = mark_correct(row.get("method_a_correct"))
        b_correct = mark_correct(row.get("method_b_correct"))
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(row['section'])}</td>"
            f"<td>{html.escape(row['item_id'])}</td>"
            f"<td>{html.escape(row.get('expected') or '')}</td>"
            f"<td>{html.escape(row.get('method_a_value') or '')} <span>{a_correct}</span></td>"
            f"<td>{html.escape(row.get('method_a_status') or '')}</td>"
            f"<td>{html.escape(row.get('method_b_value') or '')} <span>{b_correct}</span></td>"
            f"<td>{html.escape(row.get('method_b_status') or '')}</td>"
            "</tr>"
        )

    return f"""
<section class="sheet" id="{html.escape(image_id)}">
  <header>
    <h2>{html.escape(image_id)}</h2>
    <div class="meta">{len(rows)} differences</div>
  </header>
  <div class="panels">
    <figure>
      <figcaption>{label_a}</figcaption>
      <div class="image-wrap">
        <img src="{image_a}" alt="{label_a} {html.escape(image_id)}" loading="lazy">
        {overlay_svg(rows, "method_a", template, specs_by_item, specs_by_choice)}
      </div>
    </figure>
    <figure>
      <figcaption>{label_b}</figcaption>
      <div class="image-wrap">
        <img src="{image_b}" alt="{label_b} {html.escape(image_id)}" loading="lazy">
        {overlay_svg(rows, "method_b", template, specs_by_item, specs_by_choice)}
      </div>
    </figure>
  </div>
  <table>
    <thead>
      <tr>
        <th>Section</th><th>Item</th><th>GT</th>
        <th>{label_a}</th><th>Status</th>
        <th>{label_b}</th><th>Status</th>
      </tr>
    </thead>
    <tbody>{''.join(table_rows)}</tbody>
  </table>
</section>
"""


def mark_correct(value: str | None) -> str:
    if value == "True":
        return "✓"
    if value == "False":
        return "×"
    return ""


def build_index(sheet_ids: list[str]) -> str:
    links = "".join(f'<a href="#{html.escape(image_id)}">{html.escape(image_id)}</a>' for image_id in sheet_ids)
    return f'<nav class="sheet-nav">{links}</nav>'


def build_html(
    *,
    comparison_dir: Path,
    output_path: Path,
    summary: dict,
    differences: list[dict],
    records_a: dict[str, dict],
    records_b: dict[str, dict],
    template: dict,
    max_sheets: int,
) -> str:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in differences:
        grouped[row["image_id"]].append(row)

    sheet_ids = sorted(grouped, key=lambda image_id: (-len(grouped[image_id]), image_id))
    if max_sheets > 0:
        sheet_ids = sheet_ids[:max_sheets]

    specs_by_item, specs_by_choice = build_spec_index(template)
    cards = [
        sheet_card(
            image_id,
            grouped[image_id],
            records_a[image_id],
            records_b[image_id],
            summary["labels"],
            output_path,
            template,
            specs_by_item,
            specs_by_choice,
        )
        for image_id in sheet_ids
        if image_id in records_a and image_id in records_b
    ]

    pair = summary["pairwise"]["overall"]
    label_a = html.escape(summary["labels"]["method_a"])
    label_b = html.escape(summary["labels"]["method_b"])
    css = """
* { box-sizing: border-box; }
body { margin: 0; font-family: Arial, sans-serif; background: #f5f7fb; color: #1f2937; }
header.hero { padding: 20px 24px; background: #111827; color: white; position: sticky; top: 0; z-index: 5; }
h1 { margin: 0 0 8px; font-size: 22px; }
.stats { display: flex; gap: 14px; flex-wrap: wrap; color: #d1d5db; font-size: 14px; }
.sheet-nav { display: flex; gap: 8px; overflow-x: auto; padding: 10px 24px; background: white; border-bottom: 1px solid #d9dee8; position: sticky; top: 82px; z-index: 4; }
.sheet-nav a { white-space: nowrap; color: #1d4ed8; text-decoration: none; font-size: 13px; padding: 4px 8px; border: 1px solid #bfdbfe; border-radius: 4px; }
main { padding: 18px 24px 40px; }
.sheet { margin: 0 0 24px; background: white; border: 1px solid #d9dee8; border-radius: 6px; overflow: hidden; }
.sheet > header { display: flex; justify-content: space-between; align-items: center; padding: 12px 14px; background: #eef2ff; border-bottom: 1px solid #d9dee8; }
.sheet h2 { margin: 0; font-size: 17px; }
.meta { color: #4b5563; font-size: 13px; }
.panels { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; padding: 14px; align-items: start; }
figure { margin: 0; min-width: 0; }
figcaption { font-weight: 700; margin: 0 0 6px; }
.image-wrap { position: relative; width: 100%; border: 1px solid #cfd6e3; background: #fff; }
.image-wrap img { display: block; width: 100%; height: auto; }
.overlay { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
.group { fill: rgba(245, 158, 11, 0.10); stroke: rgba(245, 158, 11, 0.95); stroke-width: 3; }
.selected { fill: rgba(239, 68, 68, 0.22); stroke: rgba(220, 38, 38, 1); stroke-width: 5; }
.bubble-label { fill: #b91c1c; font-size: 20px; font-weight: 700; paint-order: stroke; stroke: white; stroke-width: 4px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 7px 9px; border-top: 1px solid #e5e7eb; text-align: left; vertical-align: top; }
th { background: #f9fafb; color: #374151; position: sticky; top: 122px; }
td span { font-weight: 700; margin-left: 4px; }
@media (max-width: 1000px) { .panels { grid-template-columns: 1fr; } .sheet-nav { top: 105px; } }
"""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Side-by-side Comparison</title>
  <style>{css}</style>
</head>
<body>
  <header class="hero">
    <h1>Side-by-side Comparison</h1>
    <div class="stats">
      <span>{label_a} vs {label_b}</span>
      <span>same {pair['same']}/{pair['total']}</span>
      <span>different {pair['different']}</span>
      <span>source {html.escape(comparison_dir.as_posix())}</span>
    </div>
  </header>
  {build_index(sheet_ids)}
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    comparison_dir = resolve_project_path(args.comparison_dir)
    output_path = resolve_project_path(args.output) if args.output else comparison_dir / "side_by_side.html"
    template = load_template(resolve_project_path(args.template))
    summary = json.loads((comparison_dir / "comparison_summary.json").read_text(encoding="utf-8"))
    differences = load_csv(comparison_dir / "differences.csv")

    method_a_path = resolve_project_path(summary["inputs"]["method_a"])
    method_b_path = resolve_project_path(summary["inputs"]["method_b"])
    records_a = load_records(method_a_path)
    records_b = load_records(method_b_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_html(
            comparison_dir=comparison_dir,
            output_path=output_path,
            summary=summary,
            differences=differences,
            records_a=records_a,
            records_b=records_b,
            template=template,
            max_sheets=args.max_sheets,
        ),
        encoding="utf-8",
    )
    print(f"output={output_path}")
    print(f"sheets={len({row['image_id'] for row in differences})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
