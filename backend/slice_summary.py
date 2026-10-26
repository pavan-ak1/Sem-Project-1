"""Slice-level summaries for final kidney CT segmentation masks.

The functions in this module operate only on already-generated CPU masks. They
never change a prediction mask or run model inference.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.ndimage import generate_binary_structure, label


CLASS_IDS: Mapping[str, int] = {
    "background": 0,
    "kidney": 1,
    "tumor": 2,
    "cyst": 3,
}
DISCLAIMER = (
    "This report summarizes automated segmentation model predictions. It is not a clinical diagnosis "
    "and should not be used as a substitute for evaluation by a qualified medical professional."
)

REPORT_COLUMNS = [
    "case_id", "slice_index", "kidney_detected", "tumor_detected", "cyst_detected",
    "background_pixels", "kidney_pixels", "tumor_pixels", "cyst_pixels",
    "background_percentage", "kidney_percentage", "tumor_percentage", "cyst_percentage",
    "kidney_connected_regions_raw", "tumor_connected_regions_raw", "cyst_connected_regions_raw",
    "kidney_connected_regions", "tumor_connected_regions", "cyst_connected_regions",
    "total_connected_regions", "kidney_component_sizes", "tumor_component_sizes", "cyst_component_sizes",
    "minimum_component_pixels", "connectivity", "quality_flags", "summary", "report_disclaimer",
]


def _validate_mask(mask: np.ndarray) -> np.ndarray:
    """Return an integer 2D mask or raise a clear error for unusable input."""
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D predicted mask, got shape {array.shape}.")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("Predicted mask must have a numeric dtype.")
    if not np.isfinite(array).all():
        raise ValueError("Predicted mask contains non-finite values.")
    if np.issubdtype(array.dtype, np.floating) and not np.equal(array, np.floor(array)).all():
        raise ValueError("Predicted mask must contain integer class IDs, not probabilities or logits.")
    return array.astype(np.int64, copy=False)


def _component_stats(mask: np.ndarray, class_id: int, structure: np.ndarray, minimum_size: int) -> tuple[int, list[int]]:
    labeled, raw_count = label(mask == class_id, structure=structure)
    if raw_count == 0:
        return 0, []
    sizes = np.bincount(labeled.ravel(), minlength=raw_count + 1)[1:]
    return int(raw_count), [int(size) for size in sizes if size >= minimum_size]


def summarize_slice_mask(
    mask: np.ndarray,
    case_id: str,
    slice_index: int,
    *,
    minimum_component_pixels: int = 1,
    connectivity: int = 8,
) -> dict[str, Any]:
    """Create a faithful report row for one final predicted segmentation mask.

    Component-size filtering is reporting-only: pixel counts and ``mask`` are
    deliberately left untouched. Detection flags reflect qualifying components.
    """
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("case_id must be a non-empty string.")
    if isinstance(slice_index, bool) or not isinstance(slice_index, (int, np.integer)):
        raise ValueError("slice_index must be an integer.")
    if minimum_component_pixels < 1:
        raise ValueError("minimum_component_pixels must be at least 1.")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be either 4 or 8 for 2D masks.")

    pred = _validate_mask(mask)
    total_pixels = int(pred.size)
    structure = generate_binary_structure(2, 1 if connectivity == 4 else 2)
    known_ids = set(CLASS_IDS.values())
    unexpected_ids = sorted(int(value) for value in np.unique(pred) if int(value) not in known_ids)
    flags: list[str] = []
    if total_pixels == 0:
        flags.append("empty_mask")
    if unexpected_ids:
        flags.append("unexpected_class_ids:" + ",".join(map(str, unexpected_ids)))

    pixel_counts = {name: int(np.count_nonzero(pred == class_id)) for name, class_id in CLASS_IDS.items()}
    percentages = {
        name: round((count / total_pixels * 100.0) if total_pixels else 0.0, 6)
        for name, count in pixel_counts.items()
    }

    components: dict[str, tuple[int, list[int]]] = {}
    for name, class_id in CLASS_IDS.items():
        if name == "background":
            continue
        if pixel_counts[name] == 0:
            components[name] = (0, [])
        else:
            components[name] = _component_stats(pred, class_id, structure, minimum_component_pixels)
    qualifying_counts = {name: len(sizes) for name, (_, sizes) in components.items()}
    kidney_detected = qualifying_counts["kidney"] > 0
    tumor_detected = qualifying_counts["tumor"] > 0
    cyst_detected = qualifying_counts["cyst"] > 0
    if (tumor_detected or cyst_detected) and not kidney_detected:
        flags.append("tumor_or_cyst_class_predicted_without_qualifying_kidney")

    retained_names = [
        label_name for label_name, detected in (
            ("kidney", kidney_detected), ("tumor", tumor_detected), ("cyst", cyst_detected)
        ) if detected
    ]
    if not retained_names:
        summary = "No foreground region retained."
    elif tumor_detected and not kidney_detected:
        summary = "Tumor-class region predicted without a qualifying kidney region."
    elif cyst_detected and not kidney_detected:
        summary = "Cyst-class region predicted without a qualifying kidney region."
    elif kidney_detected and not tumor_detected and not cyst_detected:
        summary = "Kidney region predicted; no tumor or cyst class retained."
    else:
        labels = " and ".join(f"{name} class" for name in retained_names)
        summary = f"{labels.capitalize()} predicted."

    row: dict[str, Any] = {
        "case_id": case_id.strip(), "slice_index": int(slice_index),
        "kidney_detected": kidney_detected, "tumor_detected": tumor_detected, "cyst_detected": cyst_detected,
        "minimum_component_pixels": int(minimum_component_pixels), "connectivity": connectivity,
        "quality_flags": ";".join(flags), "summary": summary, "report_disclaimer": DISCLAIMER,
    }
    for name in CLASS_IDS:
        row[f"{name}_pixels"] = pixel_counts[name]
        row[f"{name}_percentage"] = percentages[name]
    for name, (raw_count, sizes) in components.items():
        row[f"{name}_connected_regions_raw"] = raw_count
        row[f"{name}_connected_regions"] = len(sizes)
        row[f"{name}_component_sizes"] = json.dumps(sizes)
    row["total_connected_regions"] = sum(qualifying_counts.values())
    return row


def build_case_slice_summaries(
    masks: np.ndarray | Iterable[np.ndarray],
    case_ids: Sequence[str],
    slice_indices: Sequence[int],
    *,
    minimum_component_pixels: int = 1,
    connectivity: int = 8,
) -> list[dict[str, Any]]:
    """Summarize masks, preserving their explicit case/slice identities and order."""
    mask_list = list(masks)
    if not (len(mask_list) == len(case_ids) == len(slice_indices)):
        raise ValueError("masks, case_ids, and slice_indices must have the same length.")
    identities = list(zip(case_ids, slice_indices))
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate (case_id, slice_index) pairs are not valid report inputs.")
    rows = [
        summarize_slice_mask(mask, str(case_id), slice_index,
                             minimum_component_pixels=minimum_component_pixels, connectivity=connectivity)
        for mask, case_id, slice_index in zip(mask_list, case_ids, slice_indices)
    ]
    return sorted(rows, key=lambda row: (row["case_id"], row["slice_index"]))


def write_case_slice_summary_csv(rows: Sequence[Mapping[str, Any]], output_dir: Path, case_id: str) -> Path:
    """Write a stable, per-case CSV plus metadata sidecar and return the CSV path."""
    if not case_id or not str(case_id).strip():
        raise ValueError("case_id is required for CSV export.")
    ordered_rows = sorted(rows, key=lambda row: (str(row["case_id"]), int(row["slice_index"])))
    if any(str(row["case_id"]) != str(case_id) for row in ordered_rows):
        raise ValueError("A per-case CSV may contain rows for only its requested case_id.")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / f"{case_id}_slice_summary.csv"
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(ordered_rows)
        metadata_path = output_path / f"{case_id}_slice_summary_metadata.json"
        metadata_path.write_text(json.dumps({"case_id": case_id, "disclaimer": DISCLAIMER}, indent=2), encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Could not write slice summary report in {output_path}: {exc}") from exc
    return csv_path
