import csv

import numpy as np
import pytest

from backend.slice_summary import build_case_slice_summaries, summarize_slice_mask, write_case_slice_summary_csv


def test_all_background_slice_has_no_foreground_detection():
    row = summarize_slice_mask(np.zeros((3, 4), dtype=np.uint8), "case_a", 2)
    assert not row["kidney_detected"] and not row["tumor_detected"] and not row["cyst_detected"]
    assert row["background_percentage"] == 100.0
    assert row["total_connected_regions"] == 0


@pytest.mark.parametrize(("class_id", "field"), [(1, "kidney"), (2, "tumor"), (3, "cyst")])
def test_single_class_detection_and_percentages(class_id, field):
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:2, :2] = class_id
    row = summarize_slice_mask(mask, "case_a", 0)
    assert row[f"{field}_detected"]
    assert row[f"{field}_pixels"] == 4
    assert row[f"{field}_percentage"] == 25.0


def test_kidney_and_tumor_components_are_reported_separately():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[0:2, 0:2] = 1
    mask[3, 3] = 2
    mask[4, 4] = 2  # diagonal: same 8-connected tumor component
    row = summarize_slice_mask(mask, "case_a", 1, connectivity=8)
    assert row["kidney_connected_regions"] == 1
    assert row["tumor_connected_regions"] == 1
    assert row["total_connected_regions"] == 2


def test_kidney_and_cyst_are_both_reported():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:2, :2] = 1
    mask[3, 3] = 3
    row = summarize_slice_mask(mask, "case_a", 4)
    assert row["kidney_detected"] and row["cyst_detected"]
    assert row["kidney_connected_regions"] == 1
    assert row["cyst_connected_regions"] == 1


def test_small_components_are_retained_in_pixels_but_filtered_only_for_reporting():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[0, 0] = 1
    mask[2:4, 2:4] = 1
    row = summarize_slice_mask(mask, "case_a", 1, minimum_component_pixels=2)
    assert row["kidney_pixels"] == 5
    assert row["kidney_connected_regions_raw"] == 2
    assert row["kidney_connected_regions"] == 1
    assert row["kidney_detected"]


def test_tumor_without_kidney_is_preserved_and_flagged_for_review():
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 1] = 2
    row = summarize_slice_mask(mask, "case_a", 1)
    assert row["tumor_detected"] and not row["kidney_detected"]
    assert row["tumor_pixels"] == 1
    assert "without_qualifying_kidney" in row["quality_flags"]


def test_noncontiguous_shuffled_slices_sort_per_case_and_csv_export(tmp_path):
    masks = [np.zeros((2, 2), dtype=np.uint8) for _ in range(3)]
    masks[0][0, 0] = 3
    rows = build_case_slice_summaries(masks, ["case_b", "case_a", "case_a"], [9, 20, 3])
    assert [(row["case_id"], row["slice_index"]) for row in rows] == [
        ("case_a", 3), ("case_a", 20), ("case_b", 9)
    ]
    case_a_rows = [row for row in rows if row["case_id"] == "case_a"]
    csv_path = write_case_slice_summary_csv(case_a_rows, tmp_path, "case_a")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        exported = list(csv.DictReader(handle))
    assert [int(row["slice_index"]) for row in exported] == [3, 20]
    assert "automated segmentation model predictions" in exported[0]["report_disclaimer"]
