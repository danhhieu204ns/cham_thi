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
from PIL import Image, ImageDraw

from . import exam_code, part1, part2, part3, sbd
from .bubble_crops import crop_and_prelabel
from .grid_refinement import grid_block_for_spec, refine_grid_specs
from .group_relabel import relabel_groups_by_score
from .identity import crop_output_path, relabel_identity_groups
from .local_alignment import align_sheet_blocks_locally
from .markers import detect_registration_markers
from .template import canonical_size
from .warp import warp_from_markers


@dataclass(frozen=True)
class ExtractionThresholds:
    blank: float = 0.025
    filled: float = 0.08
    answer_margin: float = 0.025
    identity_filled: float = 0.07
    identity_margin: float = 0.03


@dataclass(frozen=True)
class BubbleModelSettings:
    filled_threshold: float = 0.90
    margin_threshold: float = 0.10
    batch_size: int = 256


def build_all_specs(template: dict) -> list[dict]:
    return [
        *part1.build_part1_specs(template),
        *sbd.build_specs(template),
        *exam_code.build_specs(template),
        *part2.build_specs(template),
        *part3.build_specs(template),
    ]


def alignment_block_for_spec(spec: dict) -> str:
    if spec.get("alignment_block"):
        return str(spec["alignment_block"])
    if spec.get("section") == "identity":
        return "identity"
    return str(spec.get("section") or "unknown")


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
    global_warped, matrix = warp_from_markers(
        image=image,
        source_markers=source_markers,
        target_markers=target_markers,
        output_size=output_size,
    )
    warped, local_alignment = align_sheet_blocks_locally(global_warped, template)
    return warped, {
        "status": "ok",
        "method": "global_homography_plus_local_blocks",
        "markers_found": sorted(source_markers),
        "markers": {
            key: [round(value[0], 2), round(value[1], 2)]
            for key, value in sorted(source_markers.items())
        },
        "matrix": np.round(matrix, 6).tolist(),
        "local_alignment": local_alignment,
    }


def warped_to_pil(warped: np.ndarray, template: dict) -> Image.Image:
    image = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)).convert("RGB")
    target_size = canonical_size(template)
    if image.size != target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)
    return image


def save_bbox_overlay(
    image: Image.Image,
    crop_records: Iterable[dict],
    output_path: Path,
    *,
    project_root: Path | None = None,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    colors = {
        "filled": (34, 197, 94),
        "ambiguous": (245, 158, 11),
        "invalid": (239, 68, 68),
        "blank": (59, 130, 246),
    }

    for record in crop_records:
        x1, y1, x2, y2 = (int(value) for value in record["bbox"])
        prelabel = str(record.get("prelabel", "blank"))
        red, green, blue = colors.get(prelabel, (107, 114, 128))
        width = 3 if prelabel == "filled" else 2 if prelabel in {"ambiguous", "invalid"} else 1
        alpha = 52 if prelabel != "blank" else 18
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(red, green, blue, 235),
            fill=(red, green, blue, alpha),
            width=width,
        )
        if prelabel in {"filled", "ambiguous", "invalid"}:
            label = str(record.get("label") or record.get("choice") or "")
            if label:
                draw.text((x1, max(0, y1 - 11)), label[:8], fill=(red, green, blue, 255))

    legend = [
        ("filled", colors["filled"]),
        ("ambiguous", colors["ambiguous"]),
        ("invalid", colors["invalid"]),
        ("blank", colors["blank"]),
    ]
    legend_x, legend_y = 18, 18
    for index, (label, color) in enumerate(legend):
        y = legend_y + index * 22
        draw.rectangle([legend_x, y, legend_x + 14, y + 14], fill=(*color, 185))
        draw.text((legend_x + 20, y - 1), label, fill=(*color, 255))

    annotated = Image.alpha_composite(base, overlay).convert("RGB")
    annotated.save(output_path, quality=92)
    if project_root is not None:
        return path_for_json(output_path, project_root)
    return output_path.as_posix()


def crop_specs_from_image(
    image: Image.Image,
    specs: Iterable[dict],
    *,
    sheet_meta: dict,
    thresholds: ExtractionThresholds,
    alignment_blocks: dict[str, dict] | None = None,
    grid_refinement_blocks: dict[str, dict] | None = None,
    bubble_classifier: object | None = None,
    bubble_model_settings: BubbleModelSettings | None = None,
    output_dir: Path | None = None,
    project_root: Path | None = None,
) -> list[dict]:
    crop_records: list[dict] = []
    crops_by_spec_id = {}
    alignment_blocks = alignment_blocks or {}
    grid_refinement_blocks = grid_refinement_blocks or {}
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
                "fill_score": round(crop_result.fill_score, 6),
                "ink_ratio_inside": round(crop_result.ink_ratio_inside, 6),
                "background_noise": round(crop_result.background_noise, 6),
                "darkness_contrast": round(crop_result.darkness_contrast, 6),
                "connected_component_score": round(crop_result.connected_component_score, 6),
                "prelabel": crop_result.prelabel,
                "prelabel_source": "adaptive_rule",
                "crop_path": crop_path_value,
                **alignment_metadata_for_spec(spec, alignment_blocks),
                **grid_refinement_metadata_for_spec(spec, grid_refinement_blocks),
            }
        )
        crops_by_spec_id[spec["spec_id"]] = crop_result.crop

    if bubble_classifier is not None:
        settings = bubble_model_settings or BubbleModelSettings()
        apply_bubble_classifier(
            crop_records,
            crops_by_spec_id,
            bubble_classifier=bubble_classifier,
            batch_size=settings.batch_size,
        )
        relabel_groups_by_score(
            crop_records,
            filled_threshold=settings.filled_threshold,
            margin_threshold=settings.margin_threshold,
            score_key="filled_probability",
            output_dir=output_dir,
            project_root=project_root,
        )
    else:
        answer_records = [
            record
            for record in crop_records
            if record.get("section") in {"part1", "part2", "part3"}
        ]
        relabel_groups_by_score(
            answer_records,
            filled_threshold=thresholds.filled,
            margin_threshold=thresholds.answer_margin,
            score_key="fill_score",
            output_dir=output_dir,
            project_root=project_root,
        )

    return crop_records


def alignment_metadata_for_spec(spec: dict, alignment_blocks: dict[str, dict]) -> dict:
    block_name = alignment_block_for_spec(spec)
    block_info = alignment_blocks.get(block_name, {})
    pre_correction = block_info.get("pre_correction") or {}
    post_correction = block_info.get("post_correction") or {}
    return {
        "alignment_block": block_name,
        "alignment_status": block_info.get("status"),
        "alignment_confidence": block_info.get("confidence"),
        "alignment_method": block_info.get("method"),
        "alignment_marker_count": block_info.get("marker_count"),
        "marker_pre_residual_px": pre_correction.get("rms_px"),
        "marker_residual_px": post_correction.get("rms_px"),
        "marker_max_residual_px": post_correction.get("max_px"),
    }


def grid_refinement_metadata_for_spec(spec: dict, grid_refinement_blocks: dict[str, dict]) -> dict:
    block_name = grid_block_for_spec(spec)
    block_info = grid_refinement_blocks.get(block_name, {})
    return {
        "grid_refinement_block": block_name,
        "grid_refinement_status": block_info.get("status"),
        "grid_refinement_confidence": block_info.get("confidence"),
        "grid_refinement_method": block_info.get("method"),
        "grid_refinement_matched_count": block_info.get("matched_count"),
        "grid_refinement_inlier_count": block_info.get("inlier_count"),
        "grid_residual_px": block_info.get("grid_residual_px"),
        "grid_max_residual_px": block_info.get("max_residual_px"),
        "grid_refinement_decode_allowed": block_info.get("decode_allowed"),
    }


def apply_bubble_classifier(
    crop_records: list[dict],
    crops_by_spec_id: dict[str, Image.Image],
    *,
    bubble_classifier: object,
    batch_size: int,
) -> None:
    items = [
        (str(record["spec_id"]), crops_by_spec_id[str(record["spec_id"])])
        for record in crop_records
    ]
    predictions = bubble_classifier.predict_images(items, batch_size=batch_size)
    predictions_by_spec_id = {prediction.image_path: prediction for prediction in predictions}

    for record in crop_records:
        prediction = predictions_by_spec_id[str(record["spec_id"])]
        record.update(
            {
                "model_label": prediction.label,
                "model_confidence": round(prediction.confidence, 6),
                "filled_probability": (
                    round(prediction.filled_probability, 6)
                    if prediction.filled_probability is not None
                    else None
                ),
                "blank_probability": (
                    round(prediction.blank_probability, 6)
                    if prediction.blank_probability is not None
                    else None
                ),
                "model_argmax_label": prediction.argmax_label,
                "model_argmax_confidence": round(prediction.argmax_confidence, 6),
                "model_filled_threshold": prediction.threshold,
                "prelabel_source": "bubble_classifier_group",
            }
        )


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


def sheet_status_from_warp(warp_info: dict) -> str:
    grid_status = warp_info.get("grid_refinement", {}).get("status")
    if grid_status == "ok":
        return "ok"
    if grid_status is not None:
        return "need_review"

    statuses = {
        str(status)
        for status in (
            warp_info.get("status"),
            warp_info.get("local_alignment", {}).get("status"),
        )
        if status is not None
    }
    if statuses <= {"ok"}:
        return "ok"
    return "need_review"


def _numeric_confidence(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _min_confidence(values: Iterable[object]) -> float | None:
    numeric_values = [
        value
        for value in (_numeric_confidence(item) for item in values)
        if value is not None
    ]
    if not numeric_values:
        return None
    return round(min(numeric_values), 6)


def _alignment_confidence(warp_info: dict) -> float | None:
    block_values = [
        block.get("confidence")
        for block in warp_info.get("local_alignment", {}).get("blocks", {}).values()
    ]
    return _min_confidence(block_values)


def _grid_confidence(warp_info: dict) -> float | None:
    block_values = [
        block.get("confidence")
        for block in warp_info.get("grid_refinement", {}).get("blocks", {}).values()
    ]
    return _min_confidence(block_values)


def _decoded_item_confidence(
    identity: dict,
    part1_section: dict,
    part2_section: dict,
    part3_section: dict,
) -> float | None:
    values: list[object] = []
    values.extend(field.get("confidence") for field in identity.values())
    for section in (part1_section, part2_section, part3_section):
        for answer in section.get("answers", {}).values():
            if answer.get("status") != "blank":
                values.append(answer.get("confidence"))
    return _min_confidence(values)


def _review_item_count(
    identity: dict,
    part1_section: dict,
    part2_section: dict,
    part3_section: dict,
) -> int:
    total = len(part1_section.get("review_items", []))
    for section in (part2_section, part3_section):
        counts = section.get("counts", {})
        total += int(counts.get("need_review", 0))
        total += int(counts.get("multi_mark", 0))
    for field in identity.values():
        if field.get("status") != "accepted":
            total += 1
    return total


def build_confidence_summary(
    *,
    sheet_status: str,
    warp_info: dict,
    identity: dict,
    part1_section: dict,
    part2_section: dict,
    part3_section: dict,
) -> dict:
    alignment_confidence = _alignment_confidence(warp_info)
    grid_confidence = _grid_confidence(warp_info)
    decoded_confidence = _decoded_item_confidence(identity, part1_section, part2_section, part3_section)
    sheet_confidence = _min_confidence(
        [alignment_confidence, grid_confidence, decoded_confidence]
    )
    review_count = _review_item_count(identity, part1_section, part2_section, part3_section)
    return {
        "sheet_confidence": sheet_confidence,
        "alignment_confidence": alignment_confidence,
        "grid_confidence": grid_confidence,
        "decoded_item_confidence": decoded_confidence,
        "review_item_count": review_count,
        "auto_pass": sheet_status == "ok" and review_count == 0,
    }


def extract_sheet(
    input_path: Path,
    template: dict,
    *,
    project_root: Path,
    thresholds: ExtractionThresholds | None = None,
    bubble_classifier: object | None = None,
    bubble_model_settings: BubbleModelSettings | None = None,
    crop_output_dir: Path | None = None,
    warped_output_path: Path | None = None,
    bbox_overlay_path: Path | None = None,
) -> dict:
    thresholds = thresholds or ExtractionThresholds()
    bubble_model_settings = bubble_model_settings or BubbleModelSettings()
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

    specs = build_all_specs(template)
    refined_specs, grid_refinement_info = refine_grid_specs(image, specs, template)
    warp_info["grid_refinement"] = grid_refinement_info
    sheet_status = sheet_status_from_warp(warp_info)

    crop_records = crop_specs_from_image(
        image,
        refined_specs,
        sheet_meta=sheet_meta,
        thresholds=thresholds,
        alignment_blocks=warp_info.get("local_alignment", {}).get("blocks", {}),
        grid_refinement_blocks=grid_refinement_info.get("blocks", {}),
        bubble_classifier=bubble_classifier,
        bubble_model_settings=bubble_model_settings,
        output_dir=crop_output_dir,
        project_root=project_root,
    )
    if bubble_classifier is None:
        relabel_identity_groups(
            crop_records,
            filled_threshold=thresholds.identity_filled,
            margin_threshold=thresholds.identity_margin,
            score_key="fill_score",
            output_dir=crop_output_dir,
            project_root=project_root if crop_output_dir is not None else None,
        )

    if bbox_overlay_path is not None:
        bbox_overlay_path = (
            bbox_overlay_path
            if bbox_overlay_path.is_absolute()
            else project_root / bbox_overlay_path
        )
        warp_info["bbox_overlay_path"] = save_bbox_overlay(
            image,
            crop_records,
            bbox_overlay_path,
            project_root=project_root,
        )

    records_by_section: dict[str, list[dict]] = defaultdict(list)
    for record in crop_records:
        record.pop("_crop_name", None)
        records_by_section[record["section"]].append(record)

    extra = decode_identity_and_written_sections(crop_records, template, sheet_meta)
    part1_section = decode_part1_section(records_by_section["part1"])
    identity_section = extra["identity"]
    part2_section = extra["part2"]
    part3_section = extra["part3"]
    confidence = build_confidence_summary(
        sheet_status=sheet_status,
        warp_info=warp_info,
        identity=identity_section,
        part1_section=part1_section,
        part2_section=part2_section,
        part3_section=part3_section,
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": sheet_status,
        "template_id": template.get("template_id"),
        **sheet_meta,
        "warp": warp_info,
        "thresholds": {
            "blank": thresholds.blank,
            "filled": thresholds.filled,
            "answer_margin": thresholds.answer_margin,
            "identity_filled": thresholds.identity_filled,
            "identity_margin": thresholds.identity_margin,
            "bubble_classifier": (
                {
                    "filled": bubble_model_settings.filled_threshold,
                    "margin": bubble_model_settings.margin_threshold,
                    "batch_size": bubble_model_settings.batch_size,
                    "model_path": str(getattr(bubble_classifier, "model_path", "unknown")),
                }
                if bubble_classifier is not None
                else None
            ),
        },
        "summary": {
            "sbd": extra["identity"]["sbd"]["value"],
            "sbd_status": extra["identity"]["sbd"]["status"],
            "exam_code": extra["identity"]["exam_code"]["value"],
            "exam_code_status": extra["identity"]["exam_code"]["status"],
            "alignment_status": warp_info.get("local_alignment", {}).get("status"),
            "grid_refinement_status": warp_info.get("grid_refinement", {}).get("status"),
            "sheet_confidence": confidence["sheet_confidence"],
            "review_item_count": confidence["review_item_count"],
            "crop_count": len(crop_records),
        },
        "confidence": confidence,
        "identity": identity_section,
        "part1": part1_section,
        "part2": part2_section,
        "part3": part3_section,
    }
