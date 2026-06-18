"""Full extraction pipeline for a single answer sheet."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from . import exam_code, part1, part2, part3, sbd
from .bubble_crops import crop_and_prelabel
from .identity import crop_output_path, relabel_identity_groups
from .markers import detect_registration_markers
from .template import canonical_size
from .warp import warp_from_markers


@dataclass(frozen=True)
class ExtractionThresholds:
    blank: float = 0.025
    filled: float = 0.08
    identity_filled: float = 0.07
    identity_margin: float = 0.03


def build_all_specs(template: dict) -> list[dict]:
    return [
        *part1.build_part1_specs(template),
        *sbd.build_specs(template),
        *exam_code.build_specs(template),
        *part2.build_specs(template),
        *part3.build_specs(template),
    ]


def group_by_item(records: Iterable[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["item_id"]].append(record)
    return groups


def path_for_json(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def image_id_from_input(path: Path, project_root: Path) -> str:
    try:
        raw = path.resolve().relative_to(project_root.resolve()).with_suffix("").as_posix()
    except ValueError:
        raw = path.with_suffix("").name
    image_id = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_")
    return image_id or "sheet"


def warp_sheet_image(
    input_path: Path,
    template: dict,
) -> tuple[np.ndarray, dict]:
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"cv2.imread returned None for {input_path}")

    output_size = canonical_size(template)
    target_markers = template["registration_marks"]["centers"]
    marker_tolerance = float(template["registration_marks"].get("tolerance_px", 18))
    source_markers = detect_registration_markers(
        image,
        target_markers=target_markers,
        tolerance_px=max(35.0, marker_tolerance * 2.0),
    )
    warped, matrix = warp_from_markers(
        image=image,
        source_markers=source_markers,
        target_markers=target_markers,
        output_size=output_size,
    )
    return warped, {
        "status": "ok",
        "markers_found": sorted(source_markers),
        "markers": {
            key: [round(value[0], 2), round(value[1], 2)]
            for key, value in sorted(source_markers.items())
        },
        "matrix": np.round(matrix, 6).tolist(),
    }


def warped_to_pil(warped: np.ndarray, template: dict) -> Image.Image:
    image = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)).convert("RGB")
    target_size = canonical_size(template)
    if image.size != target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)
    return image


def crop_specs_from_image(
    image: Image.Image,
    specs: Iterable[dict],
    *,
    sheet_meta: dict,
    thresholds: ExtractionThresholds,
    output_dir: Path | None = None,
    project_root: Path | None = None,
) -> list[dict]:
    crop_records: list[dict] = []
    for spec in specs:
        bbox = tuple(int(value) for value in spec["bbox"])
        crop_result = crop_and_prelabel(
            image,
            bbox,
            blank_threshold=thresholds.blank,
            filled_threshold=thresholds.filled,
        )

        crop_name = f"{sheet_meta['image_id']}__{spec['spec_id']}.jpg"
        crop_path_value = None
        if output_dir is not None:
            crop_path = crop_output_path(
                output_dir,
                spec["section"],
                crop_result.prelabel,
                crop_name,
            )
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            crop_result.crop.save(crop_path, quality=92)
            if project_root is not None:
                crop_path_value = path_for_json(crop_path, project_root)
            else:
                crop_path_value = crop_path.as_posix()

        crop_records.append(
            {
                **spec,
                "_crop_name": crop_name,
                **sheet_meta,
                "darkness_score": round(crop_result.darkness_score, 6),
                "prelabel": crop_result.prelabel,
                "crop_path": crop_path_value,
            }
        )
    return crop_records


def decode_identity_and_written_sections(crop_records: list[dict], template: dict, sheet_meta: dict) -> dict:
    records_by_section: dict[str, list[dict]] = defaultdict(list)
    for record in crop_records:
        records_by_section[record["section"]].append(record)

    identity_groups = group_by_item(records_by_section["identity"])
    part2_groups = group_by_item(records_by_section["part2"])
    part3_groups = group_by_item(records_by_section["part3"])

    return {
        **sheet_meta,
        "identity": {
            "sbd": sbd.decode(identity_groups, template),
            "exam_code": exam_code.decode(identity_groups, template),
        },
        "part2": part2.decode(part2_groups),
        "part3": part3.decode(part3_groups),
    }


def decode_part1_section(crop_records: list[dict]) -> dict:
    decoded = part1.decode_part1_records(crop_records)
    if not decoded:
        return {"part": "I", "answers": {}, "review_items": [], "counts": {}}
    sheet = decoded[0]
    return {
        "part": "I",
        "answers": sheet["answers"],
        "review_items": sheet["review_items"],
        "counts": sheet["counts"],
    }


def summarize_extra_sections(decoded_sheets: list[dict], crop_records: list[dict]) -> dict:
    identity_counts: Counter[str] = Counter()
    part2_counts: Counter[str] = Counter()
    part3_counts: Counter[str] = Counter()
    crop_counts: Counter[str] = Counter()
    for record in crop_records:
        crop_counts[f"{record['section']}:{record['prelabel']}"] += 1
    for sheet in decoded_sheets:
        identity_counts[sheet["identity"]["sbd"]["status"]] += 1
        identity_counts[sheet["identity"]["exam_code"]["status"]] += 1
        part2_counts.update(sheet["part2"]["counts"])
        part3_counts.update(sheet["part3"]["counts"])
    return {
        "sheet_count": len(decoded_sheets),
        "crop_count": len(crop_records),
        "identity_status_counts": dict(sorted(identity_counts.items())),
        "part2_status_counts": dict(sorted(part2_counts.items())),
        "part3_status_counts": dict(sorted(part3_counts.items())),
        "crop_prelabel_counts": dict(sorted(crop_counts.items())),
    }


def extract_sheet(
    input_path: Path,
    template: dict,
    *,
    project_root: Path,
    thresholds: ExtractionThresholds | None = None,
    crop_output_dir: Path | None = None,
    warped_output_path: Path | None = None,
) -> dict:
    thresholds = thresholds or ExtractionThresholds()
    input_path = input_path if input_path.is_absolute() else project_root / input_path
    input_path = input_path.resolve()
    image_id = image_id_from_input(input_path, project_root)
    source_path = path_for_json(input_path, project_root)

    warped, warp_info = warp_sheet_image(input_path, template)
    if warped_output_path is not None:
        warped_output_path = (
            warped_output_path
            if warped_output_path.is_absolute()
            else project_root / warped_output_path
        )
        warped_output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(warped_output_path), warped, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        warp_info["warped_path"] = path_for_json(warped_output_path, project_root)

    image = warped_to_pil(warped, template)
    sheet_meta = {
        "image_id": image_id,
        "source_path": source_path,
        "input_path": warp_info.get("warped_path", source_path),
    }
    if crop_output_dir is not None and not crop_output_dir.is_absolute():
        crop_output_dir = project_root / crop_output_dir

    crop_records = crop_specs_from_image(
        image,
        build_all_specs(template),
        sheet_meta=sheet_meta,
        thresholds=thresholds,
        output_dir=crop_output_dir,
        project_root=project_root,
    )
    relabel_identity_groups(
        crop_records,
        filled_threshold=thresholds.identity_filled,
        margin_threshold=thresholds.identity_margin,
        output_dir=crop_output_dir,
        project_root=project_root if crop_output_dir is not None else None,
    )

    records_by_section: dict[str, list[dict]] = defaultdict(list)
    for record in crop_records:
        record.pop("_crop_name", None)
        records_by_section[record["section"]].append(record)

    extra = decode_identity_and_written_sections(crop_records, template, sheet_meta)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "ok",
        "template_id": template.get("template_id"),
        **sheet_meta,
        "warp": warp_info,
        "thresholds": {
            "blank": thresholds.blank,
            "filled": thresholds.filled,
            "identity_filled": thresholds.identity_filled,
            "identity_margin": thresholds.identity_margin,
        },
        "summary": {
            "sbd": extra["identity"]["sbd"]["value"],
            "sbd_status": extra["identity"]["sbd"]["status"],
            "exam_code": extra["identity"]["exam_code"]["value"],
            "exam_code_status": extra["identity"]["exam_code"]["status"],
            "crop_count": len(crop_records),
        },
        "identity": extra["identity"],
        "part1": decode_part1_section(records_by_section["part1"]),
        "part2": extra["part2"],
        "part3": extra["part3"],
    }
