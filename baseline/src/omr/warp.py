"""Homography-based sheet warping."""

from __future__ import annotations

import cv2
import numpy as np


CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")


def warp_from_markers(
    image: np.ndarray,
    source_markers: dict[str, tuple[float, float]],
    target_markers: dict[str, list[int]],
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    missing = [corner for corner in CORNER_ORDER if corner not in source_markers]
    if missing:
        raise ValueError(f"missing source markers: {missing}")

    matched_names = [name for name in target_markers if name in source_markers]
    if len(matched_names) > 4:
        src = np.array([source_markers[name] for name in matched_names], dtype=np.float32)
        dst = np.array([target_markers[name] for name in matched_names], dtype=np.float32)
        matrix, _ = cv2.findHomography(src, dst, method=0)
        if matrix is not None:
            warped = cv2.warpPerspective(image, matrix, output_size, flags=cv2.INTER_LINEAR)
            return warped, matrix

    src = np.array([source_markers[corner] for corner in CORNER_ORDER], dtype=np.float32)
    dst = np.array([target_markers[corner] for corner in CORNER_ORDER], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, output_size, flags=cv2.INTER_LINEAR)
    return warped, matrix

