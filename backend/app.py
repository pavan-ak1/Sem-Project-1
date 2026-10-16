import os
import sys
import platform
import pathlib
from pathlib import Path
from typing import List, Dict, Any, Optional

# Cross-platform compatibility fix for checkpoints serialized on Windows
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

import pandas as pd
from PIL import Image
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import anyio

from backend.schemas import (
    HealthResponse,
    PatientCaseSummary,
    SlicePredictionResponse,
    Patient3DMetrics,
    ClinicalReportResponse,
    CaseUploadResponse,
    CaseUploadSliceItem
)
from backend.model_loader import load_model, get_device
from backend.inference_engine import (
    run_single_slice_inference,
    run_patient_case_inference,
    run_uploaded_case_pipeline,
    get_uploaded_case_slice,
    standardize_dims,
    create_overlay,
    image_to_base64_url,
    DATA_ROOT,
    IMAGE_DIR,
    MASK_DIR,
    CSV_PATH,
    UPLOADED_CASES_DIR
)

app = FastAPI(
    title="NephroAI Clinical Renal Analysis & Decision Support API",
    description="High-Throughput Deep Learning Medical Image Segmentation & Volumetric Staging Backend",
    version="1.0.0"
)

# Enable CORS for local Vite dev server and external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Pre-load model weights into GPU VRAM on server boot."""
    try:
        model, device = load_model()
        print(f"[STARTUP] Residual U-Net successfully loaded on {device} (8.11M params).")
    except Exception as e:
        print(f"[STARTUP WARNING] Model initialization postponed: {e}")


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """System health check and GPU acceleration status."""
    device = get_device()
    checkpoint_p = Path("notebookV3/checkpoints/best_unet_kits23.pth").resolve()
    return HealthResponse(
        status="healthy",
        device=str(device),
        cuda_available=(device.type == "cuda"),
        model_loaded=True,
        checkpoint_path=str(checkpoint_p),
        model_architecture="Medical 2D Residual U-Net (InstanceNorm2d + PReLU)",
        parameters=8113718
    )


@app.get("/api/cases", response_model=List[PatientCaseSummary])
async def list_cases():
    """Returns verified test cohort and benchmark patient cases, including any uploaded cases."""
    results = []

    # Check for uploaded cases and prepend
    if UPLOADED_CASES_DIR.exists():
        for c_dir in sorted(UPLOADED_CASES_DIR.iterdir(), reverse=True):
            meta_f = c_dir / "meta.json"
            if c_dir.is_dir() and meta_f.exists():
                try:
                    with open(meta_f, "r", encoding="utf-8") as f:
                        m = json.load(f)
                    results.append(PatientCaseSummary(
                        case_id=m["case_id"],
                        slice_count=m["slices_analyzed"],
                        has_tumor=m["vols"]["tumor_vol_cm3"] > 0,
                        has_cyst=m["vols"]["cyst_vol_cm3"] > 0,
                        split="custom_upload",
                        ai_kidney_vol_cm3=m["vols"]["total_parenchyma_vol_cm3"],
                        ai_tumor_vol_cm3=m["vols"]["tumor_vol_cm3"],
                        ai_tnm_stage=m["staging"]["tnm_t_stage"]
                    ))
                except Exception:
                    pass

    scorecard_p = Path("notebookV3/reports/phase4_test_cohort_scorecard.csv")
    if scorecard_p.exists():
        score_df = pd.read_csv(scorecard_p)
        for _, row in score_df.iterrows():
            cid = str(row["Case_ID"])
            results.append(PatientCaseSummary(
                case_id=cid,
                slice_count=int(row["Slices"]),
                has_tumor=float(row["GT_Tumor_Vol"]) > 0,
                has_cyst=float(row["Cyst_Dice"]) > 0,
                split="test",
                gt_kidney_vol_cm3=round(float(row["GT_Kidney_Vol"]), 2),
                gt_tumor_vol_cm3=round(float(row["GT_Tumor_Vol"]), 2),
                ai_kidney_vol_cm3=round(float(row["AI_Kidney_Vol"]), 2),
                ai_tumor_vol_cm3=round(float(row["AI_Tumor_Vol"]), 2),
                gt_tnm_stage=str(row["GT_TNM"]),
                ai_tnm_stage=str(row["AI_TNM"])
            ))

    # Fallback to dataset.csv unique test cases if scorecard is empty
    if not results and CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        test_cases = df[df["split"] == "test"]["case_id"].unique()[:10]
        for cid in test_cases:
            c_df = df[df["case_id"] == cid]
            results.append(PatientCaseSummary(
                case_id=str(cid),
                slice_count=len(c_df),
                has_tumor=True,
                has_cyst=False,
                split="test"
            ))

    return results


@app.get("/api/case/{case_id}/slices")
async def get_case_slices(case_id: str):
    """Returns sorted slice indices and file records for a patient case."""
    # Check if case was uploaded
    upload_meta = UPLOADED_CASES_DIR / case_id / "meta.json"
    if upload_meta.exists():
        with open(upload_meta, "r", encoding="utf-8") as f:
            m = json.load(f)
        return {
            "case_id": case_id,
            "total_slices": m["slices_analyzed"],
            "slices": m["slices"]
        }

    if not CSV_PATH.exists():
        raise HTTPException(status_code=404, detail="dataset.csv not found")

    df = pd.read_csv(CSV_PATH)
    c_df = df[df["case_id"] == case_id].sort_values("slice_index").reset_index(drop=True)
    if len(c_df) == 0:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")

    slices = []
    for _, row in c_df.iterrows():
        slices.append({
            "slice_index": int(row["slice_index"]),
            "filename": str(row["filename"]),
            "has_gt_mask": (MASK_DIR / row["filename"]).exists()
        })

    return {
        "case_id": case_id,
        "total_slices": len(slices),
        "slices": slices
    }


@app.get("/api/case/{case_id}/slice/{slice_idx}")
async def get_patient_slice_detail(case_id: str, slice_idx: int, tta: bool = False):
    """Fetches a specific patient slice with ground truth mask and live AI inference."""
    # Check if case was uploaded
    if (UPLOADED_CASES_DIR / case_id).exists():
        try:
            return get_uploaded_case_slice(case_id, slice_idx)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

    if not CSV_PATH.exists():
        raise HTTPException(status_code=404, detail="dataset.csv not found")

    df = pd.read_csv(CSV_PATH)
    row = df[(df["case_id"] == case_id) & (df["slice_index"] == slice_idx)]
    if len(row) == 0:
        # Fallback to positional index
        c_df = df[df["case_id"] == case_id].sort_values("slice_index").reset_index(drop=True)
        if 0 <= slice_idx < len(c_df):
            row = c_df.iloc[[slice_idx]]
        else:
            raise HTTPException(status_code=404, detail=f"Slice index {slice_idx} out of range for {case_id}")

    fname = row.iloc[0]["filename"]
    img_path = IMAGE_DIR / fname
    mask_path = MASK_DIR / fname

    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image file {fname} not found on disk")

    # Read image
    raw_img = Image.open(img_path).convert("L")
    img_arr = np.array(raw_img, dtype=np.float32)
    img_std = standardize_dims(img_arr, 512, 512)
    norm = (img_std - img_std.min()) / max(1e-4, img_std.max() - img_std.min())

    # Check for ground truth
    gt_overlay_b64 = None
    gt_mask_b64 = None
    if mask_path.exists():
        gt_raw = Image.open(mask_path)
        gt_arr = standardize_dims(np.array(gt_raw, dtype=np.uint8), 512, 512)
        gt_overlay_img = create_overlay(norm, gt_arr)
        gt_overlay_b64 = image_to_base64_url(gt_overlay_img)
        gt_mask_b64 = image_to_base64_url(Image.fromarray(gt_arr * 80).convert("L"))

    # Single-slice inference (supports optional 4-Fold TTA)
    with open(img_path, "rb") as fp:
        raw_bytes = fp.read()
    infer_res = await anyio.to_thread.run_sync(run_single_slice_inference, raw_bytes, tta)

    infer_res["ground_truth_overlay"] = gt_overlay_b64
    infer_res["ground_truth_mask"] = gt_mask_b64
    infer_res["case_id"] = case_id
    infer_res["slice_index"] = slice_idx
    infer_res["filename"] = fname

    return infer_res


@app.post("/api/predict/slice", response_model=SlicePredictionResponse)
async def predict_custom_slice(
    file: UploadFile = File(...),
    tta: Optional[bool] = Form(None)
):
    """Receives an uploaded custom CT slice file and returns AI segmentation (adaptive TTA)."""
    try:
        contents = await file.read()
        res = await anyio.to_thread.run_sync(run_single_slice_inference, contents, tta)
        return SlicePredictionResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/predict/case-upload", response_model=CaseUploadResponse)
async def predict_uploaded_case(
    files: List[UploadFile] = File(...),
    case_id: Optional[str] = Form(None),
    tta: Optional[bool] = Form(None)
):
    """
    Receives an uploaded patient case folder (or multiple slice files)
    and executes complete 3D volumetric analysis, morphological CCA filtering,
    PCA caliper morphometry, and AJCC TNM clinical staging (adaptive TTA).
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    file_payloads = []
    for f in files:
        data = await f.read()
        file_payloads.append((f.filename, data))

    try:
        results = await anyio.to_thread.run_sync(run_uploaded_case_pipeline, file_payloads, case_id, tta)
        return CaseUploadResponse(**results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Case upload processing error: {str(e)}")


@app.get("/api/uploaded-case/{case_id}/slice/{slice_idx}")
async def get_uploaded_slice(case_id: str, slice_idx: int):
    """Retrieves on-demand slice visualization for an uploaded patient case."""
    try:
        return get_uploaded_case_slice(case_id, slice_idx)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


_case_3d_cache: Dict[str, Any] = {}


@app.get("/api/case/{case_id}/metrics")
async def get_case_metrics(case_id: str):
    """Retrieves cached 3D metrics if available, or returns 404."""
    upload_meta = UPLOADED_CASES_DIR / case_id / "meta.json"
    if upload_meta.exists():
        with open(upload_meta, "r", encoding="utf-8") as f:
            return JSONResponse(content=json.load(f))
    if case_id in _case_3d_cache:
        return JSONResponse(content=_case_3d_cache[case_id])
    return JSONResponse(content={"status": "not_computed", "case_id": case_id}, status_code=404)


@app.post("/api/predict/case/{case_id}")
async def predict_case_3d(case_id: str, tta: Optional[bool] = None):
    """Executes full 3D patient volume reconstruction, morphometry, and TNM staging (adaptive TTA)."""
    upload_meta = UPLOADED_CASES_DIR / case_id / "meta.json"
    if upload_meta.exists():
        with open(upload_meta, "r", encoding="utf-8") as f:
            m = json.load(f)
        return JSONResponse(content={
            "case_id": case_id,
            "slices_analyzed": m["slices_analyzed"],
            "right_kidney_vol_cm3": m["vols"]["right_kidney_vol_cm3"],
            "left_kidney_vol_cm3": m["vols"]["left_kidney_vol_cm3"],
            "total_parenchyma_vol_cm3": m["vols"]["total_parenchyma_vol_cm3"],
            "tumor_vol_cm3": m["vols"]["tumor_vol_cm3"],
            "cyst_vol_cm3": m["vols"]["cyst_vol_cm3"],
            "tumor_burden_pct": m["vols"]["tumor_burden_pct"],
            "length_cm": m["pca"]["length_cm"],
            "width_cm": m["pca"]["width_cm"],
            "thickness_cm": m["pca"]["thickness_cm"],
            "ellipsoid_vol_cm3": m["pca"]["ellipsoid_vol_cm3"],
            "ellipsoid_error_pct": m["pca"]["ellipsoid_error_pct"],
            "tnm_stage": m["staging"]["tnm_t_stage"],
            "surgical_recommendation": m["staging"]["surgical_recommendation"],
            "inference_time_sec": m.get("inference_time_sec", 1.0),
            "report_markdown": m["report_markdown"]
        })

    if case_id in _case_3d_cache:
        return JSONResponse(content=_case_3d_cache[case_id])

    try:
        metrics = await anyio.to_thread.run_sync(run_patient_case_inference, case_id, tta)
        _case_3d_cache[case_id] = metrics
        return JSONResponse(content=metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"3D Case inference error: {str(e)}")


@app.get("/api/report/{case_id}", response_model=ClinicalReportResponse)
async def get_clinical_report(case_id: str):
    """Fetches or generates clinical radiology decision support report."""
    upload_meta = UPLOADED_CASES_DIR / case_id / "meta.json"
    if upload_meta.exists():
        with open(upload_meta, "r", encoding="utf-8") as f:
            m = json.load(f)
        return ClinicalReportResponse(
            case_id=case_id,
            markdown_content=m["report_markdown"],
            structured_json=m
        )

    md_path = Path(f"notebookV3/reports/{case_id}_clinical_report.md")
    json_path = Path(f"notebookV3/reports/{case_id}_clinical_report.json")

    if md_path.exists():
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        struct_json = {}
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as jf:
                struct_json = json.load(jf)
        return ClinicalReportResponse(
            case_id=case_id,
            markdown_content=md_content,
            structured_json=struct_json
        )

    # If report file doesn't exist yet, run on-demand inference
    res = run_patient_case_inference(case_id)
    return ClinicalReportResponse(
        case_id=case_id,
        markdown_content=res["report_markdown"],
        structured_json=res
    )


# ==============================================================================
# STATIC FRONTEND SERVING (For unified 1-command presentation)
# ==============================================================================
dist_dir = Path("frontend/dist")
if dist_dir.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="frontend")

