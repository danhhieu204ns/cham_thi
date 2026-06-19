"""Registration marker detection for scanned answer sheets."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MarkerCandidate:
    x: int
    y: int
    w: int
    h: int
    area: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


def find_marker_candidates(image: np.ndarray) -> list[MarkerCandidate]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = gray.shape[:2]
    candidates: list[MarkerCandidate] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        if not (10 <= w <= 54 and 10 <= h <= 54):
            continue
        aspect = w / h
        if not (0.55 <= aspect <= 1.65):
            continue
        fill_ratio = area / float(w * h)
        if fill_ratio < 0.45:
            continue
        near_left = x < width * 0.12
        near_right = x > width * 0.88
        near_top = y < height * 0.12
        near_bottom = y > height * 0.82
        if not (near_left or near_right or near_top or near_bottom):
            continue
        candidates.append(MarkerCandidate(x=x, y=y, w=w, h=h, area=area))
    return sorted(candidates, key=lambda item: (item.y, item.x))


def _choose_corner(
    candidates: list[MarkerCandidate],
    *,
    width: int,
    height: int,
    corner: str,
) -> MarkerCandidate | None:
    if corner == "top_left":
        pool = [c for c in candidates if c.x < width * 0.2 and c.y < height * 0.2]
        return min(pool, key=lambda c: c.x + c.y, default=None)
    if corner == "top_right":
        pool = [c for c in candidates if c.x > width * 0.8 and c.y < height * 0.2]
        return min(pool, key=lambda c: (width - c.x) + c.y, default=None)
    if corner == "bottom_left":
        pool = [c for c in candidates if c.x < width * 0.2 and c.y > height * 0.75]
        return min(pool, key=lambda c: c.x + (height - c.y), default=None)
    if corner == "bottom_right":
        pool = [c for c in candidates if c.x > width * 0.8 and c.y > height * 0.75]
        return min(pool, key=lambda c: (width - c.x) + (height - c.y), default=None)
    raise ValueError(f"unknown corner: {corner}")


def detect_corner_markers(image: np.ndarray) -> dict[str, tuple[float, float]]:
    height, width = image.shape[:2]
    candidates = find_marker_candidates(image)
    corners: dict[str, tuple[float, float]] = {}
    for corner in ("top_left", "top_right", "bottom_left", "bottom_right"):
        candidate = _choose_corner(candidates, width=width, height=height, corner=corner)
        if candidate is not None:
            corners[corner] = candidate.center
    return corners


def detect_registration_markers(
    image: np.ndarray,
    target_markers: dict[str, list[int]],
    *,
    tolerance_px: float = 35.0,
) -> dict[str, tuple[float, float]]:
    """Match all detectable registration marks using a corner-warp bootstrap."""
    corners = detect_corner_markers(image)
    corner_order = ("top_left", "top_right", "bottom_right", "bottom_left")
    if any(corner not in corners for corner in corner_order):
        return corners

    src_corners = np.array([corners[corner] for corner in corner_order], dtype=np.float32)
    dst_corners = np.array([target_markers[corner] for corner in corner_order], dtype=np.float32)
    bootstrap = cv2.getPerspectiveTransform(src_corners, dst_corners)

    candidates = find_marker_candidates(image)
    potential_matches: list[tuple[float, str, MarkerCandidate]] = []
    for name, target in target_markers.items():
        target_x, target_y = float(target[0]), float(target[1])
        for candidate in candidates:
            point = np.array([[[candidate.center[0], candidate.center[1]]]], dtype=np.float32)
            projected = cv2.perspectiveTransform(point, bootstrap)[0][0]
            distance = float(np.hypot(projected[0] - target_x, projected[1] - target_y))
            if distance <= tolerance_px:
                potential_matches.append((distance, name, candidate))

    matches: dict[str, tuple[float, float]] = {}
    used_candidates: set[tuple[int, int, int, int]] = set()
    for _, name, candidate in sorted(potential_matches, key=lambda item: item[0]):
        candidate_key = (candidate.x, candidate.y, candidate.w, candidate.h)
        if name in matches or candidate_key in used_candidates:
            continue
        matches[name] = candidate.center
        used_candidates.add(candidate_key)

    for name, center in corners.items():
        matches.setdefault(name, center)
    return matches
