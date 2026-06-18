from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from omr.decode_answers import decode_question
from omr.geometry import bubble_bbox
from omr.group_relabel import relabel_groups_by_score
from omr.identity import decode_identity
from omr.jsonl_io import (
    read_jsonl_records,
    read_jsonl_records_if_exists,
    write_jsonl_records,
)
from omr.part2 import decode as decode_part2
from omr.part3 import compact_value, decode as decode_part3
from omr.template import part1_bubbles


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
