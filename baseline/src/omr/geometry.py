"""Geometry helpers shared by OMR template and crop code."""

from __future__ import annotations


def bubble_bbox(
    center_x: int,
    center_y: int,
    size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = size
    half_w = width // 2
    half_h = height // 2
    return center_x - half_w, center_y - half_h, center_x + half_w, center_y + half_h
