import os
import sys
import platform
import pathlib
import logging
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
import torch

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import json
import anyio

logger = logging.getLogger(__name__)

from backend.schemas import (
    HealthResponse,
    PatientCaseSummary,
    SlicePredictionResponse,
    CaseUploadResponse,
    CaseUploadSliceItem
)
from backend.model_loader import load_model, get_device, is_model_loaded
from backend.inference_engine import (
    run_single_slice_inference,
    run_patient_case_inference,
    get_patient_case_full_bundle,
    _case_full_cache,
    run_uploaded_case_pipeline,
    get_uploaded_case_slice,
    standardize_dims,
    create_overlay,
    image_to_base64_url,
    DATA_ROOT,
    IMAGE_DIR,
    MASK_DIR,
    CSV_PATH,
    UPLOADED_CASES_DIR,
    SLICE_SUMMARY_OUTPUT_DIR,
    CASE_BUNDLE_CACHE_DIR,
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
    """Pre-load model weights into GPU VRAM on server boot and warm up cuDNN kernels."""
    try:
        model, device = load_model()
        print(f"[STARTUP] Residual U-Net successfully loaded on {device} (8.11M params).")
        if device.type == "cuda":
            with torch.inference_mode():
                # Warm up batch size 4 (volume batch inference)
                dummy_batch = torch.zeros((4, 1, 512, 512), device=device)
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    _ = model(dummy_batch)
                # Warm up batch size 1 (single-slice fast interactive preview)
                dummy_single = torch.zeros((1, 1, 512, 512), device=device)
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    _ = model(dummy_single)
            torch.cuda.synchronize()
            print("[STARTUP] GPU inference engine warmed up and cuDNN convolution kernels cached (B=1, B=4).")
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
        model_loaded=is_model_loaded(),
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
                        has_tumor=any(s.get("has_tumor", False) for s in m.get("slices", [])),
                        has_cyst=any(s.get("has_cyst", False) for s in m.get("slices", [])),
                        split="custom_upload",
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

    # Always include available test cases from dataset.csv if scorecard is missing or to enrich cohort
    if CSV_PATH.exists():
        existing_ids = {r.case_id for r in results}
        df = pd.read_csv(CSV_PATH)
        test_cases = df[df["split"] == "test"]["case_id"].unique()[:15]
        for cid in test_cases:
            if str(cid) not in existing_ids:
                c_df = df[df["case_id"] == cid]
                results.append(PatientCaseSummary(
                    case_id=str(cid),
                    slice_count=len(c_df),
                    has_tumor=True,
                    has_cyst=False,
                    split="test"
                ))

    return results


@app.get("/api/case/{case_id}/full")
async def get_patient_case_full(case_id: str, tta: Optional[bool] = None):
    """
    Returns the complete patient case bundle (all slices + 3D metrics) at once
    for instant client-side slice navigation without per-slice server roundtrips.
    """
    try:
        bundle = await anyio.to_thread.run_sync(get_patient_case_full_bundle, case_id, tta)
        return JSONResponse(content=bundle)
    except Exception as e:
        logger.exception("Full case bundle failed for %s", case_id)
        raise HTTPException(status_code=500, detail=f"Case full bundle error: {str(e)}")


@app.get("/api/case/{case_id}/slices")
async def get_case_slices(case_id: str, full: bool = False):
    """Returns sorted slice indices or complete slice bundle for a patient case."""
    if full or (case_id in _case_full_cache):
        try:
            return await get_patient_case_full(case_id)
        except Exception:
            pass

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


@app.get("/api/case/{case_id}/slice-summary.csv")
async def download_slice_summary_csv(case_id: str):
    """Download the already-generated, final-mask slice summary for one case."""
    upload_meta = UPLOADED_CASES_DIR / case_id / "meta.json"
    if upload_meta.exists():
        with open(upload_meta, "r", encoding="utf-8") as handle:
            csv_value = json.load(handle).get("slice_summary_csv")
        csv_path = Path(csv_value) if csv_value else UPLOADED_CASES_DIR / case_id / "reports" / f"{case_id}_slice_summary.csv"
    else:
        csv_path = SLICE_SUMMARY_OUTPUT_DIR / f"{case_id}_slice_summary.csv"

    # If this is the first access, execute the normal case-bundle path once;
    # it writes the summary while producing the masks used by the viewer.
    if not csv_path.exists():
        await anyio.to_thread.run_sync(get_patient_case_full_bundle, case_id)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"Slice summary is unavailable for case '{case_id}'.")
    return FileResponse(csv_path, media_type="text/csv", filename=f"{case_id}_slice_summary.csv")


@app.get("/api/case/{case_id}/slice/{slice_idx}")
async def get_patient_slice_detail(case_id: str, slice_idx: int, tta: bool = False):
    """Fetches a specific patient slice with ground truth mask and live AI inference."""
    # Fast path: check in-memory full case cache or L2 disk cache
    if case_id not in _case_full_cache:
        cache_file = CASE_BUNDLE_CACHE_DIR / f"{case_id}_full.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    _case_full_cache[case_id] = json.load(f)
            except Exception:
                pass

    if case_id in _case_full_cache:
        slices = _case_full_cache[case_id].get("slices", [])
        for s in slices:
            if s.get("slice_index") == slice_idx:
                return s
        if 0 <= slice_idx < len(slices):
            return slices[slice_idx]

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
    if "slice_summary" in infer_res and isinstance(infer_res["slice_summary"], dict):
        infer_res["slice_summary"]["case_id"] = case_id
        infer_res["slice_summary"]["slice_index"] = slice_idx

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


# ==============================================================================
# STATIC FRONTEND SERVING (For unified 1-command presentation)
# ==============================================================================
dist_dir = Path("frontend/dist")
if dist_dir.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="frontend")

