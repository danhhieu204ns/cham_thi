"""Inference helpers for using a trained layout_v0 detector in extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .layout_training import MASK_CHANNELS, build_layout_model, extract_heatmap_peaks


@dataclass(frozen=True)
class LayoutDetectorSettings:
    page_threshold: float = 0.5
    grid_threshold: float = 0.5
    marker_threshold: float = 0.25
    bubble_threshold: float = 0.25
    marker_nms_radius: int = 4
    bubble_nms_radius: int = 3
    max_marker_peaks: int = 40
    max_bubble_peaks: int = 900
    marker_match_tolerance_px: float = 55.0


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device_name)


def _round_point(point: tuple[float, float]) -> list[float]:
    return [round(float(point[0]), 2), round(float(point[1]), 2)]


def _round_matrix(matrix: np.ndarray | None) -> list[list[float]] | None:
    if matrix is None:
        return None
    return np.round(matrix, 6).tolist()


def _transform_point(matrix: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    source = np.array([[[point[0], point[1]]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(source, matrix)[0][0]
    return float(transformed[0]), float(transformed[1])


def _corner_pool(
    peaks: list[dict[str, float]],
    *,
    width: int,
    height: int,
    corner: str,
    used_indexes: set[int],
) -> list[tuple[int, dict[str, float]]]:
    def unused() -> list[tuple[int, dict[str, float]]]:
        return [(index, peak) for index, peak in enumerate(peaks) if index not in used_indexes]

    candidates = unused()
    if corner == "top_left":
        filtered = [(i, p) for i, p in candidates if p["x"] < width * 0.35 and p["y"] < height * 0.35]
    elif corner == "top_right":
        filtered = [(i, p) for i, p in candidates if p["x"] > width * 0.65 and p["y"] < height * 0.35]
    elif corner == "bottom_right":
        filtered = [(i, p) for i, p in candidates if p["x"] > width * 0.65 and p["y"] > height * 0.65]
    elif corner == "bottom_left":
        filtered = [(i, p) for i, p in candidates if p["x"] < width * 0.35 and p["y"] > height * 0.65]
    else:
        raise ValueError(f"unknown corner: {corner}")
    return filtered or candidates


def _corner_score(peak: dict[str, float], *, width: int, height: int, corner: str) -> float:
    if corner == "top_left":
        return peak["x"] + peak["y"]
    if corner == "top_right":
        return (width - peak["x"]) + peak["y"]
    if corner == "bottom_right":
        return (width - peak["x"]) + (height - peak["y"])
    if corner == "bottom_left":
        return peak["x"] + (height - peak["y"])
    raise ValueError(f"unknown corner: {corner}")


def _choose_corner_peaks(
    peaks: list[dict[str, float]],
    *,
    width: int,
    height: int,
    target_markers: dict[str, list[int]],
) -> dict[str, tuple[int, dict[str, float]]]:
    corners: dict[str, tuple[int, dict[str, float]]] = {}
    used_indexes: set[int] = set()
    for corner in ("top_left", "top_right", "bottom_right", "bottom_left"):
        if corner not in target_markers:
            continue
        pool = _corner_pool(peaks, width=width, height=height, corner=corner, used_indexes=used_indexes)
        if not pool:
            continue
        index, peak = min(
            pool,
            key=lambda item: _corner_score(item[1], width=width, height=height, corner=corner),
        )
        corners[corner] = (index, peak)
        used_indexes.add(index)
    return corners


def _bootstrap_matrix(
    corner_peaks: dict[str, tuple[int, dict[str, float]]],
    target_markers: dict[str, list[int]],
) -> tuple[np.ndarray | None, list[str], str]:
    corner_order = ("top_left", "top_right", "bottom_right", "bottom_left")
    matched = [name for name in corner_order if name in corner_peaks and name in target_markers]
    if len(matched) < 3:
        return None, matched, "insufficient_corners"

    src = np.array(
        [[corner_peaks[name][1]["x"], corner_peaks[name][1]["y"]] for name in matched],
        dtype=np.float32,
    )
    dst = np.array([target_markers[name] for name in matched], dtype=np.float32)
    if len(matched) >= 4:
        matrix = cv2.getPerspectiveTransform(src[:4], dst[:4])
        return matrix, matched[:4], "homography"

    affine = cv2.getAffineTransform(src[:3], dst[:3])
    matrix = np.vstack([affine, [0.0, 0.0, 1.0]]).astype(np.float32)
    return matrix, matched[:3], "affine"


def match_marker_peaks_to_template(
    peaks: list[dict[str, float]],
    *,
    source_size: tuple[int, int],
    target_markers: dict[str, list[int]],
    tolerance_px: float,
) -> tuple[dict[str, tuple[float, float]], dict[str, Any]]:
    source_width, source_height = source_size
    if not peaks:
        return {}, {
            "status": "no_marker_peaks",
            "method": "none",
            "marker_peak_count": 0,
            "matched_corners": [],
            "matrix": None,
        }

    corner_peaks = _choose_corner_peaks(
        peaks,
        width=source_width,
        height=source_height,
        target_markers=target_markers,
    )
    matrix, matched_corners, method = _bootstrap_matrix(corner_peaks, target_markers)
    if matrix is None:
        return {}, {
            "status": "insufficient_corners",
            "method": method,
            "marker_peak_count": len(peaks),
            "matched_corners": matched_corners,
            "matrix": None,
        }

    match_tolerance = tolerance_px if len(matched_corners) >= 4 else tolerance_px * 3.0
    potential_matches: list[tuple[float, float, str, int, dict[str, float]]] = []
    for name, target in target_markers.items():
        target_x, target_y = float(target[0]), float(target[1])
        for index, peak in enumerate(peaks):
            projected_x, projected_y = _transform_point(matrix, (peak["x"], peak["y"]))
            distance = float(np.hypot(projected_x - target_x, projected_y - target_y))
            if distance <= match_tolerance:
                potential_matches.append((distance, -float(peak["score"]), name, index, peak))

    matches: dict[str, tuple[float, float]] = {}
    used_peak_indexes: set[int] = set()
    match_distances: dict[str, float] = {}
    for distance, _, name, index, peak in sorted(potential_matches, key=lambda item: (item[0], item[1])):
        if name in matches or index in used_peak_indexes:
            continue
        matches[name] = (float(peak["x"]), float(peak["y"]))
        used_peak_indexes.add(index)
        match_distances[name] = distance

    for name, (index, peak) in corner_peaks.items():
        if name in target_markers and name not in matches:
            matches[name] = (float(peak["x"]), float(peak["y"]))
            used_peak_indexes.add(index)
            match_distances[name] = 0.0

    status = "ok" if len(matches) >= 4 else "insufficient_markers"
    distances = list(match_distances.values())
    info = {
        "status": status,
        "method": method,
        "marker_peak_count": len(peaks),
        "matched_corners": matched_corners,
        "marker_match_tolerance_px": round(float(match_tolerance), 3),
        "matched_marker_count": len(matches),
        "mean_marker_match_distance_px": (
            round(float(sum(distances) / len(distances)), 3) if distances else None
        ),
        "max_marker_match_distance_px": round(float(max(distances)), 3) if distances else None,
        "matrix": _round_matrix(matrix),
    }
    return matches, info


class LayoutDetector:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        image_width: int = 0,
        image_height: int = 0,
        base_channels: int = 0,
        settings: LayoutDetectorSettings | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = choose_device(device)
        self.settings = settings or LayoutDetectorSettings()

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        config = checkpoint.get("model_config", {})
        checkpoint_size = checkpoint.get("image_size", [416, 596])
        self.image_size = (
            image_width or int(checkpoint_size[0]),
            image_height or int(checkpoint_size[1]),
        )
        model_base_channels = base_channels or int(config.get("base_channels", 24))
        self.model = build_layout_model(
            out_channels=int(config.get("out_channels", len(MASK_CHANNELS))),
            base_channels=model_base_channels,
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        output_width, output_height = self.image_size
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (output_width, output_height), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(image_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0)
        tensor = (tensor - 0.5) / 0.5
        return tensor.unsqueeze(0)

    def _scaled_peaks(
        self,
        peaks: list[tuple[float, float, float]],
        *,
        scale_x: float,
        scale_y: float,
    ) -> list[dict[str, float]]:
        return [
            {
                "x": float(x) * scale_x,
                "y": float(y) * scale_y,
                "model_x": float(x),
                "model_y": float(y),
                "score": float(score),
            }
            for x, y, score in peaks
        ]

    @torch.inference_mode()
    def detect(self, image_bgr: np.ndarray, template: dict) -> dict[str, Any]:
        source_height, source_width = image_bgr.shape[:2]
        image_tensor = self._preprocess(image_bgr).to(self.device, non_blocking=True)
        logits = self.model(image_tensor)
        probabilities = torch.sigmoid(logits)[0].detach().cpu().numpy()
        settings = self.settings

        model_marker_peaks = extract_heatmap_peaks(
            probabilities[1],
            threshold=settings.marker_threshold,
            nms_radius=settings.marker_nms_radius,
            max_peaks=settings.max_marker_peaks,
        )
        model_bubble_peaks = extract_heatmap_peaks(
            probabilities[2],
            threshold=settings.bubble_threshold,
            nms_radius=settings.bubble_nms_radius,
            max_peaks=settings.max_bubble_peaks,
        )

        model_width, model_height = self.image_size
        scale_x = source_width / float(model_width)
        scale_y = source_height / float(model_height)
        marker_peaks = self._scaled_peaks(model_marker_peaks, scale_x=scale_x, scale_y=scale_y)
        bubble_peaks = self._scaled_peaks(model_bubble_peaks, scale_x=scale_x, scale_y=scale_y)

        target_markers = template["registration_marks"]["centers"]
        markers, match_info = match_marker_peaks_to_template(
            marker_peaks,
            source_size=(source_width, source_height),
            target_markers=target_markers,
            tolerance_px=settings.marker_match_tolerance_px,
        )

        info = {
            "checkpoint": str(self.checkpoint_path),
            "device": str(self.device),
            "model_input_size": [model_width, model_height],
            "source_size": [source_width, source_height],
            "thresholds": {
                "page": settings.page_threshold,
                "grid": settings.grid_threshold,
                "markers": settings.marker_threshold,
                "bubbles": settings.bubble_threshold,
            },
            "bubble_peak_count": len(bubble_peaks),
            "markers_found": sorted(markers),
            "markers": {
                key: _round_point(value)
                for key, value in sorted(markers.items())
            },
            "marker_peaks": [
                {
                    "x": round(float(peak["x"]), 2),
                    "y": round(float(peak["y"]), 2),
                    "model_x": round(float(peak["model_x"]), 2),
                    "model_y": round(float(peak["model_y"]), 2),
                    "score": round(float(peak["score"]), 6),
                }
                for peak in marker_peaks
            ],
            "match": match_info,
        }
        return {
            "source_markers": markers,
            "info": info,
        }
