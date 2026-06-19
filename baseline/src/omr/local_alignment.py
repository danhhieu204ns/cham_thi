"""Local block alignment for marker-based OMR extraction."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .markers import match_registration_markers_by_position
from .template import canonical_size


BLOCK_DEFINITIONS = {
    "identity": {
        "regions": ("sbd", "exam_code"),
        "markers": ("top_left", "top_right", "part1_left_top", "part1_right_top"),
    },
    "part1": {
        "regions": ("part1",),
        "markers": ("part1_left_top", "part1_right_top", "part1_left_bottom", "part1_right_bottom"),
    },
    "part2": {
        "regions": ("part2",),
        "markers": ("part1_left_bottom", "part1_right_bottom", "part2_left_bottom", "part2_right_bottom"),
    },
    "part3": {
        "regions": ("part3",),
        "markers": ("part2_left_bottom", "part2_right_bottom", "bottom_left", "bottom_right"),
    },
}


@dataclass(frozen=True)
class LocalTransform:
    matrix: np.ndarray
    method: str
    status: str
    marker_names: tuple[str, ...]
    pre_errors: tuple[float, ...]
    post_errors: tuple[float, ...]
    confidence: float


def _round_point(point: tuple[float, float]) -> list[float]:
    return [round(float(point[0]), 2), round(float(point[1]), 2)]


def _round_matrix(matrix: np.ndarray) -> list[list[float]]:
    return np.round(matrix, 6).tolist()


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    reshaped = points.reshape(-1, 1, 2).astype(np.float32)
    transformed = cv2.perspectiveTransform(reshaped, matrix)
    return transformed.reshape(-1, 2)


def _error_stats(errors: tuple[float, ...]) -> dict:
    if not errors:
        return {
            "rms_px": None,
            "max_px": None,
        }
    values = np.asarray(errors, dtype=np.float32)
    return {
        "rms_px": round(float(np.sqrt(np.mean(values * values))), 3),
        "max_px": round(float(np.max(values)), 3),
    }


def estimate_local_transform(
    source_markers: dict[str, tuple[float, float]],
    target_markers: dict[str, list[int]],
    marker_names: tuple[str, ...],
    *,
    residual_threshold_px: float = 8.0,
) -> LocalTransform:
    matched_names = tuple(name for name in marker_names if name in source_markers and name in target_markers)
    identity = np.eye(3, dtype=np.float32)
    if not matched_names:
        return LocalTransform(
            matrix=identity,
            method="identity",
            status="fallback_global",
            marker_names=matched_names,
            pre_errors=(),
            post_errors=(),
            confidence=0.0,
        )

    src = np.array([source_markers[name] for name in matched_names], dtype=np.float32)
    dst = np.array([target_markers[name] for name in matched_names], dtype=np.float32)
    pre_errors = tuple(float(value) for value in np.linalg.norm(src - dst, axis=1))

    matrix: np.ndarray | None = None
    method = "identity"
    if len(matched_names) >= 4:
        matrix, _ = cv2.findHomography(src, dst, method=0)
        method = "homography"
    elif len(matched_names) >= 3:
        affine = cv2.getAffineTransform(src[:3], dst[:3])
        matrix = np.vstack([affine, [0.0, 0.0, 1.0]]).astype(np.float32)
        method = "affine"
    else:
        delta = np.median(dst - src, axis=0)
        matrix = np.array(
            [
                [1.0, 0.0, float(delta[0])],
                [0.0, 1.0, float(delta[1])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        method = "translation"

    if matrix is None:
        return LocalTransform(
            matrix=identity,
            method=method,
            status="alignment_failed",
            marker_names=matched_names,
            pre_errors=pre_errors,
            post_errors=(),
            confidence=0.0,
        )

    transformed = _transform_points(matrix.astype(np.float32), src)
    post_errors = tuple(float(value) for value in np.linalg.norm(transformed - dst, axis=1))
    post_stats = _error_stats(post_errors)
    post_rms = float(post_stats["rms_px"] or 0.0)
    marker_factor = min(1.0, len(matched_names) / 4.0)
    residual_factor = max(0.0, 1.0 - (post_rms / max(1.0, residual_threshold_px)))
    confidence = marker_factor * residual_factor

    if method in {"homography", "affine"} and post_rms <= residual_threshold_px:
        status = "ok"
    elif method == "translation":
        status = "fallback_translation"
    else:
        status = "alignment_failed"

    return LocalTransform(
        matrix=matrix.astype(np.float32),
        method=method,
        status=status,
        marker_names=matched_names,
        pre_errors=pre_errors,
        post_errors=post_errors,
        confidence=confidence,
    )


def _expanded_bbox(bbox: list[int], image_shape: tuple[int, ...], margin_px: int) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = (int(value) for value in bbox)
    return (
        max(0, x1 - margin_px),
        max(0, y1 - margin_px),
        min(width, x2 + margin_px),
        min(height, y2 + margin_px),
    )


def _copy_regions(
    destination: np.ndarray,
    source: np.ndarray,
    template: dict,
    region_names: tuple[str, ...],
    *,
    margin_px: int,
) -> dict[str, list[int]]:
    copied_regions: dict[str, list[int]] = {}
    for region_name in region_names:
        region = template.get("regions", {}).get(region_name, {})
        bbox = region.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = _expanded_bbox(bbox, destination.shape, margin_px)
        destination[y1:y2, x1:x2] = source[y1:y2, x1:x2]
        copied_regions[region_name] = [x1, y1, x2, y2]
    return copied_regions


def _transform_info(
    name: str,
    transform: LocalTransform,
    copied_regions: dict[str, list[int]],
    target_markers: dict[str, list[int]],
) -> dict:
    pre_stats = _error_stats(transform.pre_errors)
    post_stats = _error_stats(transform.post_errors)
    return {
        "status": transform.status,
        "method": transform.method,
        "confidence": round(float(transform.confidence), 3),
        "expected_markers": list(BLOCK_DEFINITIONS[name]["markers"]),
        "markers_found": list(transform.marker_names),
        "marker_count": len(transform.marker_names),
        "markers": {
            marker_name: target_markers[marker_name]
            for marker_name in transform.marker_names
            if marker_name in target_markers
        },
        "pre_correction": pre_stats,
        "post_correction": post_stats,
        "regions": copied_regions,
        "matrix": _round_matrix(transform.matrix),
    }


def _overall_status(blocks: dict[str, dict]) -> str:
    statuses = {str(info["status"]) for info in blocks.values()}
    if "alignment_failed" in statuses:
        return "need_review"
    if any(status.startswith("fallback") for status in statuses):
        return "degraded"
    return "ok"


def align_sheet_blocks_locally(
    warped: np.ndarray,
    template: dict,
) -> tuple[np.ndarray, dict]:
    """Return a canonical page composed from locally aligned regions."""
    output_size = canonical_size(template)
    target_markers = template["registration_marks"]["centers"]
    marker_tolerance = float(template["registration_marks"].get("tolerance_px", 18))
    match_tolerance = max(24.0, marker_tolerance * 2.0)
    residual_threshold = float(template["registration_marks"].get("local_residual_threshold_px", 8.0))
    region_margin = int(template.get("bubble_crop", {}).get("padding", 10)) + 8

    detected_markers = match_registration_markers_by_position(
        warped,
        target_markers=target_markers,
        tolerance_px=match_tolerance,
    )
    aligned = warped.copy()
    blocks: dict[str, dict] = {}

    for block_name, definition in BLOCK_DEFINITIONS.items():
        transform = estimate_local_transform(
            detected_markers,
            target_markers,
            tuple(definition["markers"]),
            residual_threshold_px=residual_threshold,
        )
        if transform.status == "fallback_global":
            block_aligned = warped
        else:
            block_aligned = cv2.warpPerspective(
                warped,
                transform.matrix,
                output_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        copied_regions = _copy_regions(
            aligned,
            block_aligned,
            template,
            tuple(definition["regions"]),
            margin_px=region_margin,
        )
        blocks[block_name] = _transform_info(block_name, transform, copied_regions, target_markers)

    return aligned, {
        "status": _overall_status(blocks),
        "marker_match_tolerance_px": round(match_tolerance, 3),
        "residual_threshold_px": round(residual_threshold, 3),
        "markers_found": sorted(detected_markers),
        "markers": {
            key: _round_point(value)
            for key, value in sorted(detected_markers.items())
        },
        "blocks": blocks,
    }
