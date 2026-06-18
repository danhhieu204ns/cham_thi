"""Exam-code (ma de thi) extraction."""

from __future__ import annotations

from .identity import build_identity_specs, decode_identity


FIELD = "exam_code"


def build_specs(template: dict) -> list[dict]:
    return build_identity_specs(template, FIELD)


def decode(groups: dict[str, list[dict]], template: dict) -> dict:
    digit_count = int(template["grids"][FIELD]["digit_count"])
    return decode_identity(groups, FIELD, digit_count)
