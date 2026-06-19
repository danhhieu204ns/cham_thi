"""Bubble-grid refinement after local block alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from .geometry import bubble_bbox


GRID_REVIEW_RESIDUAL_PX = 6.0
GRID_FAIL_RESIDUAL_PX = 10.0


@dataclass(frozen=True)
class GridTransform:
    matrix: np.ndarray
    method: str
    status: str
    matched_count: int
    inlier_count: int
    rms_error_px: float | None
    max_error_px: float | None
    median_shift: tuple[float, float] | None
    confidence: float
    decode_allowed: bool


def grid_block_for_spec(spec: dict) -> str:
    if spec.get("grid_block"):
        return str(spec["grid_block"])
    if spec.get("section") == "identity":
        return "identity"
    return str(spec.get("section") or "unknown")


def _round_matrix(matrix: np.ndarray) -> list[list[float]]:
    return np.round(matrix, 6).tolist()


def _round_pair(value: tuple[float, float] | None) -> list[float] | None:
    if value is None:
        return None
    return [round(float(value[0]), 3), round(float(value[1]), 3)]


def _specs_bbox(
    specs: list[dict],
    image_shape: tuple[int, ...],
    *,
    margin_px: int,
) -> tuple[int, int, int, int] | None:
    boxes = []
    for spec in specs:
        bbox = spec.get("template_bbox") or spec.get("bbox")
        if bbox:
            boxes.append(tuple(int(value) for value in bbox))
    if not boxes:
        return None
    height, width = image_shape[:2]
    return (
        max(0, min(box[0] for box in boxes) - margin_px),
        max(0, min(box[1] for box in boxes) - margin_px),
        min(width, max(box[2] for box in boxes) + margin_px),
        min(height, max(box[3] for box in boxes) + margin_px),
    )


def _detect_bubble_circles(
    image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> list[tuple[float, float, float]]:
    x1, y1, x2, y2 = bbox
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=18,
        param1=70,
        param2=24,
        minRadius=8,
        maxRadius=18,
    )
    hough_circles = []
    if circles is not None:
        hough_circles = [
            (float(center_x + x1), float(center_y + y1), float(radius))
            for center_x, center_y, radius in circles[0]
        ]

    contour_circles = _detect_contour_circles(gray, offset=(x1, y1))
    return _dedupe_circles([*hough_circles, *contour_circles])


def _detect_contour_circles(
    gray: np.ndarray,
    *,
    offset: tuple[int, int],
) -> list[tuple[float, float, float]]:
    normalized = cv2.GaussianBlur(gray, (3, 3), 0)
    threshold = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles: list[tuple[float, float, float]] = []
    offset_x, offset_y = offset
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not (35.0 <= area <= 900.0):
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if not (8 <= width <= 34 and 8 <= height <= 34):
            continue
        aspect = width / max(1, height)
        if not (0.65 <= aspect <= 1.45):
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.35:
            continue
        center_x = offset_x + x + width / 2.0
        center_y = offset_y + y + height / 2.0
        radius = (width + height) / 4.0
        circles.append((float(center_x), float(center_y), float(radius)))
    return circles


def _dedupe_circles(
    circles: list[tuple[float, float, float]],
    *,
    min_distance_px: float = 8.0,
) -> list[tuple[float, float, float]]:
    deduped: list[tuple[float, float, float]] = []
    for circle in sorted(circles, key=lambda item: item[2], reverse=True):
        if any(
            np.hypot(circle[0] - kept[0], circle[1] - kept[1]) < min_distance_px
            for kept in deduped
        ):
            continue
        deduped.append(circle)
    return deduped


def _match_anchor_points(
    specs: list[dict],
    circles: list[tuple[float, float, float]],
    *,
    max_distance_px: float,
) -> list[tuple[float, float, float, float]]:
    candidates: list[tuple[float, int, int, float, float, float, float]] = []
    for spec_index, spec in enumerate(specs):
        expected_x, expected_y = (float(value) for value in spec["center"])
        for circle_index, circle in enumerate(circles):
            detected_x, detected_y = circle[0], circle[1]
            distance = float(np.hypot(detected_x - expected_x, detected_y - expected_y))
            if distance <= max_distance_px:
                candidates.append(
                    (
                        distance,
                        spec_index,
                        circle_index,
                        expected_x,
                        expected_y,
                        detected_x,
                        detected_y,
                    )
                )

    pairs: list[tuple[float, float, float, float]] = []
    used_specs: set[int] = set()
    used_circles: set[int] = set()
    for _, spec_index, circle_index, expected_x, expected_y, detected_x, detected_y in sorted(candidates):
        if spec_index in used_specs or circle_index in used_circles:
            continue
        used_specs.add(spec_index)
        used_circles.add(circle_index)
        pairs.append((expected_x, expected_y, detected_x, detected_y))
    return pairs


def _error_stats(errors: np.ndarray) -> tuple[float, float]:
    return (
        float(np.sqrt(np.mean(errors * errors))),
        float(np.max(errors)),
    )


def estimate_grid_transform(
    pairs: Iterable[tuple[float, float, float, float]],
    *,
    expected_count: int,
    min_inlier_ratio: float = 0.35,
    ransac_threshold_px: float = 4.0,
    review_residual_threshold_px: float = GRID_REVIEW_RESIDUAL_PX,
    fail_residual_threshold_px: float = GRID_FAIL_RESIDUAL_PX,
) -> GridTransform:
    pairs = list(pairs)
    identity = np.eye(3, dtype=np.float32)
    if not pairs:
        return GridTransform(
            matrix=identity,
            method="identity",
            status="skipped",
            matched_count=0,
            inlier_count=0,
            rms_error_px=None,
            max_error_px=None,
            median_shift=None,
            confidence=0.0,
            decode_allowed=True,
        )

    src = np.asarray([[item[0], item[1]] for item in pairs], dtype=np.float32)
    dst = np.asarray([[item[2], item[3]] for item in pairs], dtype=np.float32)
    shifts = dst - src
    median_shift = tuple(float(value) for value in np.median(shifts, axis=0))

    min_inliers = max(3, int(round(expected_count * min_inlier_ratio)))
    if len(pairs) >= 3:
        affine, inliers = cv2.estimateAffine2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold_px,
            maxIters=2000,
            confidence=0.98,
            refineIters=10,
        )
        if affine is not None and inliers is not None:
            matrix = np.vstack([affine, [0.0, 0.0, 1.0]]).astype(np.float32)
            predicted = cv2.transform(src.reshape(-1, 1, 2), affine).reshape(-1, 2)
            errors = np.linalg.norm(predicted - dst, axis=1)
            inlier_mask = inliers.reshape(-1).astype(bool)
            inlier_count = int(inlier_mask.sum())
            if inlier_count >= min_inliers:
                inlier_errors = errors[inlier_mask]
                rms_error, max_error = _error_stats(inlier_errors)
                confidence = min(1.0, inlier_count / max(1, expected_count))
                status = _status_for_residual(
                    rms_error,
                    review_residual_threshold_px=review_residual_threshold_px,
                    fail_residual_threshold_px=fail_residual_threshold_px,
                )
                return GridTransform(
                    matrix=matrix,
                    method="affine",
                    status=status,
                    matched_count=len(pairs),
                    inlier_count=inlier_count,
                    rms_error_px=rms_error,
                    max_error_px=max_error,
                    median_shift=median_shift,
                    confidence=confidence if status == "ok" else min(confidence, 0.5),
                    decode_allowed=rms_error <= fail_residual_threshold_px,
                )

    matrix = np.array(
        [
            [1.0, 0.0, median_shift[0]],
            [0.0, 1.0, median_shift[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    predicted = src + np.asarray(median_shift, dtype=np.float32)
    errors = np.linalg.norm(predicted - dst, axis=1)
    rms_error, max_error = _error_stats(errors)
    if rms_error > fail_residual_threshold_px:
        status = "alignment_failed"
    elif rms_error > review_residual_threshold_px:
        status = "need_review"
    else:
        status = "fallback_translation" if len(pairs) >= 4 else "skipped"
    confidence = min(0.65, len(pairs) / max(1, expected_count))
    return GridTransform(
        matrix=matrix,
        method="translation",
        status=status,
        matched_count=len(pairs),
        inlier_count=len(pairs),
        rms_error_px=rms_error,
        max_error_px=max_error,
        median_shift=median_shift,
        confidence=confidence,
        decode_allowed=rms_error <= fail_residual_threshold_px,
    )


def _status_for_residual(
    rms_error_px: float,
    *,
    review_residual_threshold_px: float,
    fail_residual_threshold_px: float,
) -> str:
    if rms_error_px > fail_residual_threshold_px:
        return "alignment_failed"
    if rms_error_px > review_residual_threshold_px:
        return "need_review"
    return "ok"


def _transform_center(matrix: np.ndarray, center: list[int] | tuple[int, int]) -> tuple[int, int]:
    point = np.asarray([[[float(center[0]), float(center[1])]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, matrix.astype(np.float32))[0][0]
    return int(round(float(transformed[0]))), int(round(float(transformed[1])))


def _transform_info(transform: GridTransform, circle_count: int) -> dict:
    return {
        "status": transform.status,
        "method": transform.method,
        "confidence": round(float(transform.confidence), 3),
        "decode_allowed": transform.decode_allowed,
        "circle_count": circle_count,
        "matched_count": transform.matched_count,
        "inlier_count": transform.inlier_count,
        "rms_error_px": (
            round(float(transform.rms_error_px), 3)
            if transform.rms_error_px is not None
            else None
        ),
        "max_error_px": (
            round(float(transform.max_error_px), 3)
            if transform.max_error_px is not None
            else None
        ),
        "grid_residual_px": (
            round(float(transform.rms_error_px), 3)
            if transform.rms_error_px is not None
            else None
        ),
        "max_residual_px": (
            round(float(transform.max_error_px), 3)
            if transform.max_error_px is not None
            else None
        ),
        "median_shift_px": _round_pair(transform.median_shift),
        "matrix": _round_matrix(transform.matrix),
    }


def refine_grid_specs(
    image: Image.Image,
    specs: Iterable[dict],
    template: dict,
    *,
    max_match_distance_px: float | None = None,
    review_residual_threshold_px: float | None = None,
    fail_residual_threshold_px: float | None = None,
) -> tuple[list[dict], dict]:
    specs = [dict(spec) for spec in specs]
    settings = template.get("grid_refinement", {})
    max_match_distance_px = float(
        settings.get("max_match_distance_px", 32.0)
        if max_match_distance_px is None
        else max_match_distance_px
    )
    review_residual_threshold_px = float(
        settings.get("review_residual_threshold_px", GRID_REVIEW_RESIDUAL_PX)
        if review_residual_threshold_px is None
        else review_residual_threshold_px
    )
    fail_residual_threshold_px = float(
        settings.get("fail_residual_threshold_px", GRID_FAIL_RESIDUAL_PX)
        if fail_residual_threshold_px is None
        else fail_residual_threshold_px
    )
    image_bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    specs_by_block: dict[str, list[dict]] = {}
    for spec in specs:
        specs_by_block.setdefault(grid_block_for_spec(spec), []).append(spec)

    transforms: dict[str, GridTransform] = {}
    block_info: dict[str, dict] = {}
    margin_px = int(max(template["bubble_crop"]["size"])) + int(template.get("bubble_crop", {}).get("padding", 10))
    for block_name, block_specs in specs_by_block.items():
        bbox = _specs_bbox(block_specs, image_bgr.shape, margin_px=margin_px)
        if bbox is None:
            continue
        circles = _detect_bubble_circles(image_bgr, bbox)
        pairs = _match_anchor_points(
            block_specs,
            circles,
            max_distance_px=max_match_distance_px,
        )
        transform = estimate_grid_transform(
            pairs,
            expected_count=len(block_specs),
            review_residual_threshold_px=review_residual_threshold_px,
            fail_residual_threshold_px=fail_residual_threshold_px,
        )
        transforms[block_name] = transform
        block_info[block_name] = _transform_info(transform, len(circles))

    crop_size = tuple(int(value) for value in template["bubble_crop"]["size"])
    refined_specs: list[dict] = []
    for spec in specs:
        block_name = grid_block_for_spec(spec)
        transform = transforms.get(block_name)
        refined = dict(spec)
        refined["template_center"] = list(spec["center"])
        refined["template_bbox"] = list(spec["bbox"])
        refined["grid_refinement_block"] = block_name
        if transform is not None and transform.status in {"ok", "need_review", "fallback_translation"}:
            center_x, center_y = _transform_center(transform.matrix, spec["center"])
            refined["center"] = [center_x, center_y]
            refined["bbox"] = list(bubble_bbox(center_x, center_y, crop_size))
        refined_specs.append(refined)

    status_values = {info["status"] for info in block_info.values()}
    if not status_values:
        status = "skipped"
    elif status_values <= {"ok"}:
        status = "ok"
    elif status_values <= {"skipped"}:
        status = "skipped"
    else:
        status = "need_review"
    return refined_specs, {
        "status": status,
        "max_match_distance_px": round(max_match_distance_px, 3),
        "review_residual_threshold_px": round(review_residual_threshold_px, 3),
        "fail_residual_threshold_px": round(fail_residual_threshold_px, 3),
        "blocks": block_info,
    }
