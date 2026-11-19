from backend.schemas import CaseUploadResponse, CaseUploadSliceItem


def test_case_upload_schema_preserves_slice_summary_and_pixel_counts():
    slice_item = CaseUploadSliceItem(
        slice_index=0,
        filename="slice_0000.png",
        kidney_area_cm2=14.5,
        tumor_area_cm2=2.1,
        cyst_area_cm2=0.0,
        has_kidney=True,
        has_tumor=True,
        has_cyst=False,
        pixel_counts={"background": 250000, "kidney": 2300, "tumor": 340, "cyst": 0},
        slice_summary={
            "case_id": "test_case",
            "slice_index": 0,
            "kidney_detected": True,
            "tumor_detected": True,
            "cyst_detected": False,
            "kidney_percentage": 0.88,
            "tumor_percentage": 0.13,
            "cyst_percentage": 0.0,
            "background_percentage": 98.99,
        },
    )

    data = {
        "case_id": "test_case",
        "slices_analyzed": 1,
        "best_preview_slice": 0,
        "inference_time_sec": 1.25,
        "tta_enabled": False,
        "kidney_absence_filter": None,
        "slices": [slice_item],
        "key_preview_slices": [],
        "slice_summary": [
            {
                "case_id": "test_case",
                "slice_index": 0,
                "kidney_detected": True,
                "tumor_detected": True,
                "cyst_detected": False,
            }
        ],
        "slice_summary_csv": "/tmp/test_case_slice_summary.csv",
    }

    response = CaseUploadResponse(**data)
    dumped = response.model_dump() if hasattr(response, "model_dump") else response.dict()

    assert dumped["case_id"] == "test_case"
    assert dumped["slices"][0]["slice_summary"]["kidney_detected"] is True
    assert dumped["slices"][0]["pixel_counts"]["kidney"] == 2300
    assert dumped["slice_summary"][0]["slice_index"] == 0
    assert dumped["slice_summary_csv"] == "/tmp/test_case_slice_summary.csv"

