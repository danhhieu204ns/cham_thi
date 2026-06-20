from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from omr.bubble_crops import bubble_features
from omr.decode_answers import decode_question
from omr.geometry import bubble_bbox
from omr.grid_refinement import estimate_grid_transform
from omr.group_relabel import relabel_groups_by_score
from omr.identity import decode_identity
from omr.jsonl_io import (
    read_jsonl_records,
    read_jsonl_records_if_exists,
    write_jsonl_records,
)
from omr.local_alignment import estimate_local_transform
from omr.layout_training import layout_loss
from omr.markers import detect_registration_markers
from omr.part2 import decode as decode_part2
from omr.part3 import compact_value, decode as decode_part3
from omr.part1 import build_part1_specs
from omr.template import part1_bubbles
from omr.warp import warp_from_markers


def bubble_record(choice: str, prelabel: str, score: float) -> dict:
    return {
        "question_id": "I_001",
        "question_number": 1,
        "choice": choice,
        "prelabel": prelabel,
        "darkness_score": score,
        "crop_path": f"crops/I_001_{choice}.jpg",
    }


def group_record(item_id: str, choice: str, prelabel: str, score: float) -> dict:
    return {
        "item_id": item_id,
        "question_id": item_id,
        "question_number": 1,
        "choice": choice,
        "label": choice,
        "prelabel": prelabel,
        "darkness_score": score,
        "crop_path": None,
    }


class GeometryTests(unittest.TestCase):
    def test_bubble_bbox_is_centered_on_even_crop_size(self) -> None:
        self.assertEqual(bubble_bbox(30, 20, (12, 10)), (24, 15, 36, 25))


class BubbleScoringTests(unittest.TestCase):
    def test_bubble_features_score_center_fill_above_blank(self) -> None:
        blank = Image.new("RGB", (36, 36), "white")
        filled = Image.new("RGB", (36, 36), "white")
        draw = ImageDraw.Draw(filled)
        draw.ellipse([10, 10, 26, 26], fill="black")

        blank_features = bubble_features(blank)
        filled_features = bubble_features(filled)

        self.assertLess(blank_features["fill_score"], 0.01)
        self.assertGreater(filled_features["fill_score"], blank_features["fill_score"] + 0.1)
        self.assertGreater(filled_features["connected_component_score"], 0.1)


class LayoutLossTests(unittest.TestCase):
    def test_pos_weight_increases_positive_target_loss(self) -> None:
        logits = torch.zeros((1, 2, 1, 1), dtype=torch.float32)
        targets = torch.tensor([[[[1.0]], [[0.0]]]], dtype=torch.float32)

        baseline = layout_loss(
            logits,
            targets,
            pos_weights=(1.0, 1.0),
            dice_weight=0.0,
            mse_weight=0.0,
        )
        weighted = layout_loss(
            logits,
            targets,
            pos_weights=(5.0, 1.0),
            dice_weight=0.0,
            mse_weight=0.0,
        )

        self.assertGreater(float(weighted), float(baseline))


class LocalAlignmentTests(unittest.TestCase):
    def test_estimate_local_transform_uses_translation_fallback_for_two_markers(self) -> None:
        target = {
            "a": [10, 10],
            "b": [30, 10],
        }
        source = {
            "a": (8, 13),
            "b": (28, 13),
        }

        transform = estimate_local_transform(source, target, ("a", "b"))

        self.assertEqual(transform.method, "translation")
        self.assertEqual(transform.status, "fallback_translation")
        self.assertAlmostEqual(float(transform.matrix[0, 2]), 2.0)
        self.assertAlmostEqual(float(transform.matrix[1, 2]), -3.0)

    def test_estimate_local_transform_reports_missing_markers_as_global_fallback(self) -> None:
        transform = estimate_local_transform({}, {"a": [10, 10]}, ("a",))

        self.assertEqual(transform.status, "fallback_global")
        self.assertEqual(transform.confidence, 0.0)


class GridRefinementTests(unittest.TestCase):
    def test_estimate_grid_transform_fits_shifted_grid(self) -> None:
        pairs = [
            (10.0, 10.0, 13.0, 8.0),
            (30.0, 10.0, 33.0, 8.0),
            (10.0, 30.0, 13.0, 28.0),
            (30.0, 30.0, 33.0, 28.0),
            (50.0, 30.0, 53.0, 28.0),
        ]

        transform = estimate_grid_transform(pairs, expected_count=5)

        self.assertEqual(transform.status, "ok")
        self.assertEqual(transform.method, "affine")
        self.assertAlmostEqual(float(transform.matrix[0, 2]), 3.0, places=4)
        self.assertAlmostEqual(float(transform.matrix[1, 2]), -2.0, places=4)

    def test_estimate_grid_transform_blocks_decode_on_large_residual(self) -> None:
        pairs = [
            (0.0, 0.0, 0.0, 0.0),
            (10.0, 0.0, 50.0, 0.0),
        ]

        transform = estimate_grid_transform(pairs, expected_count=10)

        self.assertEqual(transform.status, "alignment_failed")
        self.assertFalse(transform.decode_allowed)


class MarkerWarpTests(unittest.TestCase):
    def test_detect_registration_markers_bootstraps_from_three_corners(self) -> None:
        target_markers = {
            "top_left": [62, 74],
            "top_right": [1604, 78],
            "part1_left_top": [62, 795],
            "part1_right_top": [1602, 801],
            "bottom_left": [57, 2313],
            "bottom_right": [1600, 2317],
        }
        image = np.full((2363, 1650, 3), 255, dtype=np.uint8)
        for name, (center_x, center_y) in target_markers.items():
            if name == "bottom_right":
                continue
            cv2.rectangle(
                image,
                (center_x - 15, center_y - 15),
                (center_x + 15, center_y + 15),
                (0, 0, 0),
                thickness=-1,
            )

        detected = detect_registration_markers(image, target_markers, tolerance_px=18)

        self.assertIn("top_left", detected)
        self.assertIn("top_right", detected)
        self.assertIn("bottom_left", detected)
        self.assertIn("part1_left_top", detected)
        self.assertIn("part1_right_top", detected)
        self.assertNotIn("bottom_right", detected)

    def test_warp_from_markers_uses_affine_fallback_with_three_markers(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        source = {
            "top_left": (10.0, 10.0),
            "top_right": (90.0, 10.0),
            "bottom_left": (10.0, 90.0),
        }
        target = {
            "top_left": [0, 0],
            "top_right": [80, 0],
            "bottom_left": [0, 80],
            "bottom_right": [80, 80],
        }

        warped, matrix = warp_from_markers(image, source, target, (80, 80))

        self.assertEqual(warped.shape[:2], (80, 80))
        self.assertAlmostEqual(float(matrix[0, 0]), 1.0)
        self.assertAlmostEqual(float(matrix[1, 1]), 1.0)
        self.assertAlmostEqual(float(matrix[0, 2]), -10.0)
        self.assertAlmostEqual(float(matrix[1, 2]), -10.0)


class JsonlTests(unittest.TestCase):
    def test_jsonl_round_trip_and_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "records.jsonl"
            records = [{"b": 2, "a": 1}, {"text": "ok"}]

            write_jsonl_records(records, path)

            self.assertEqual(read_jsonl_records(path), records)
            self.assertEqual(read_jsonl_records_if_exists(Path(tmp_dir) / "missing.jsonl"), [])


class TemplateTests(unittest.TestCase):
    def test_part1_bubbles_use_shared_bbox_geometry(self) -> None:
        template = {
            "bubble_crop": {"size": [12, 10]},
            "grids": {
                "part1": {
                    "choices": ["A", "B"],
                    "columns": [
                        {
                            "question_start": 1,
                            "question_count": 2,
                            "row_y_start": 20,
                            "row_y_step": 10,
                            "choice_x": [30, 50],
                        }
                    ],
                }
            },
        }

        bubbles = list(part1_bubbles(template))

        self.assertEqual(len(bubbles), 4)
        self.assertEqual(bubbles[0].question_id, "I_001")
        self.assertEqual(bubbles[0].choice, "A")
        self.assertEqual(bubbles[0].crop_bbox, (24, 15, 36, 25))

    def test_part1_specs_include_alignment_and_subgrid_blocks(self) -> None:
        template = {
            "bubble_crop": {"size": [12, 10]},
            "grids": {
                "part1": {
                    "choices": ["A", "B"],
                    "columns": [
                        {
                            "question_start": 1,
                            "question_count": 2,
                            "row_y_start": 20,
                            "row_y_step": 10,
                            "choice_x": [30, 50],
                        }
                    ],
                }
            },
        }

        specs = build_part1_specs(template)

        self.assertEqual(specs[0]["alignment_block"], "part1")
        self.assertEqual(specs[0]["grid_block"], "part1:q001-002")


class DecodeTests(unittest.TestCase):
    def test_decode_question_accepts_single_filled_choice(self) -> None:
        result = decode_question(
            [
                bubble_record("A", "filled", 0.2),
                bubble_record("B", "blank", 0.01),
                bubble_record("C", "blank", 0.02),
                bubble_record("D", "blank", 0.03),
            ]
        )

        self.assertEqual(result["selected"], "A")
        self.assertEqual(result["status"], "accepted")
        self.assertFalse(result["need_review"])
        self.assertEqual(result["darkness_margin"], 0.17)
        self.assertEqual(result["confidence"], 1.0)

    def test_decode_question_flags_low_confidence_margin(self) -> None:
        records = [
            bubble_record("A", "filled", 0.2),
            bubble_record("B", "blank", 0.19),
            bubble_record("C", "blank", 0.01),
            bubble_record("D", "blank", 0.01),
        ]
        for record in records:
            record["group_score_key"] = "darkness_score"
            record["group_margin_threshold"] = 0.025

        result = decode_question(records)

        self.assertEqual(result["selected"], "A")
        self.assertEqual(result["status"], "need_review")
        self.assertIn("low_confidence", result["review_reasons"])

    def test_decode_question_flags_multi_mark_without_selected_answer(self) -> None:
        result = decode_question(
            [
                bubble_record("A", "filled", 0.2),
                bubble_record("B", "filled", 0.19),
                bubble_record("C", "blank", 0.02),
                bubble_record("D", "blank", 0.03),
            ]
        )

        self.assertIsNone(result["selected"])
        self.assertEqual(result["status"], "multi_mark")
        self.assertIn("multi_mark", result["review_reasons"])

    def test_decode_question_flags_alignment_failed(self) -> None:
        records = [
            bubble_record("A", "filled", 0.2),
            bubble_record("B", "blank", 0.01),
            bubble_record("C", "blank", 0.02),
            bubble_record("D", "blank", 0.03),
        ]
        records[0]["alignment_status"] = "fallback_translation"

        result = decode_question(records)

        self.assertEqual(result["selected"], "A")
        self.assertEqual(result["status"], "need_review")
        self.assertIn("alignment_failed", result["review_reasons"])

    def test_decode_question_blocks_selection_on_hard_grid_failure(self) -> None:
        records = [
            bubble_record("A", "filled", 0.2),
            bubble_record("B", "blank", 0.01),
            bubble_record("C", "blank", 0.02),
            bubble_record("D", "blank", 0.03),
        ]
        records[0]["grid_refinement_status"] = "alignment_failed"
        records[0]["grid_refinement_decode_allowed"] = False

        result = decode_question(records)

        self.assertIsNone(result["selected"])
        self.assertEqual(result["status"], "need_review")
        self.assertIn("alignment_failed", result["review_reasons"])

    def test_decode_question_accepts_when_grid_overrides_local_fallback(self) -> None:
        records = [
            bubble_record("A", "filled", 0.2),
            bubble_record("B", "blank", 0.01),
            bubble_record("C", "blank", 0.02),
            bubble_record("D", "blank", 0.03),
        ]
        for record in records:
            record["alignment_status"] = "fallback_global"
            record["grid_refinement_status"] = "ok"

        result = decode_question(records)

        self.assertEqual(result["selected"], "A")
        self.assertEqual(result["status"], "accepted")

    def test_decode_identity_compacts_digit_columns(self) -> None:
        groups = {
            "sbd_01": [group_record("sbd_01", "3", "filled", 0.2)],
            "sbd_02": [group_record("sbd_02", "7", "filled", 0.21)],
        }

        result = decode_identity(groups, "sbd", 3)

        self.assertEqual(result["value"], "37_")
        self.assertEqual(result["status"], "incomplete")

    def test_decode_part2_keeps_statement_ids(self) -> None:
        groups = {
            "II_001_a": [
                group_record("II_001_a", "T", "blank", 0.01),
                group_record("II_001_a", "F", "filled", 0.2),
            ]
        }

        result = decode_part2(groups)

        self.assertEqual(result["answers"]["II_001_a"]["selected"], "F")
        self.assertEqual(result["answers"]["II_001_a"]["statement"], "a")

    def test_decode_part3_compacts_signed_decimal_value(self) -> None:
        self.assertEqual(compact_value("-", "after_1", ["1", "2", None, None]), ("-1,2__", "-1,2"))

        groups = {
            "III_001_sign": [group_record("III_001_sign", "-", "filled", 0.2)],
            "III_001_comma": [
                group_record("III_001_comma", "after_1", "filled", 0.2),
                group_record("III_001_comma", "after_2", "blank", 0.01),
            ],
            "III_001_digit_1": [group_record("III_001_digit_1", "1", "filled", 0.2)],
            "III_001_digit_2": [group_record("III_001_digit_2", "2", "filled", 0.2)],
        }

        result = decode_part3(groups)

        self.assertEqual(result["answers"]["III_001"]["value"], "-1,2")
        self.assertEqual(result["answers"]["III_001"]["status"], "accepted")


class GroupRelabelTests(unittest.TestCase):
    def test_relabel_groups_by_score_selects_clear_top_choice(self) -> None:
        records = [
            {
                "item_id": "I_001",
                "spec_id": "I_001_A",
                "prelabel": "blank",
                "filled_probability": 0.98,
            },
            {
                "item_id": "I_001",
                "spec_id": "I_001_B",
                "prelabel": "filled",
                "filled_probability": 0.12,
            },
        ]

        relabel_groups_by_score(
            records,
            filled_threshold=0.9,
            margin_threshold=0.1,
            score_key="filled_probability",
        )

        self.assertEqual([record["prelabel"] for record in records], ["filled", "blank"])
        self.assertEqual(records[0]["group_score_key"], "filled_probability")
        self.assertEqual(records[0]["group_score_margin"], 0.86)

    def test_relabel_groups_by_score_blanks_group_below_threshold(self) -> None:
        records = [
            {
                "item_id": "I_001",
                "spec_id": "I_001_A",
                "prelabel": "filled",
                "filled_probability": 0.89,
            },
            {
                "item_id": "I_001",
                "spec_id": "I_001_B",
                "prelabel": "blank",
                "filled_probability": 0.20,
            },
        ]

        relabel_groups_by_score(
            records,
            filled_threshold=0.9,
            margin_threshold=0.1,
            score_key="filled_probability",
        )

        self.assertEqual([record["prelabel"] for record in records], ["blank", "blank"])

    def test_relabel_groups_by_score_flags_close_top_choices(self) -> None:
        records = [
            {
                "item_id": "I_001",
                "spec_id": "I_001_A",
                "prelabel": "blank",
                "filled_probability": 0.96,
            },
            {
                "item_id": "I_001",
                "spec_id": "I_001_B",
                "prelabel": "blank",
                "filled_probability": 0.91,
            },
            {
                "item_id": "I_001",
                "spec_id": "I_001_C",
                "prelabel": "blank",
                "filled_probability": 0.10,
            },
        ]

        relabel_groups_by_score(
            records,
            filled_threshold=0.9,
            margin_threshold=0.1,
            score_key="filled_probability",
        )

        self.assertEqual([record["prelabel"] for record in records], ["ambiguous", "ambiguous", "blank"])


if __name__ == "__main__":
    unittest.main()
