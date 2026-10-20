from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class HealthResponse(BaseModel):
    status: str
    device: str
    cuda_available: bool
    model_loaded: bool
    checkpoint_path: str
    model_architecture: str
    parameters: int

class PatientCaseSummary(BaseModel):
    case_id: str
    slice_count: int
    has_tumor: bool
    has_cyst: bool
    split: str
    gt_kidney_vol_cm3: Optional[float] = None
    gt_tumor_vol_cm3: Optional[float] = None
    ai_kidney_vol_cm3: Optional[float] = None
    ai_tumor_vol_cm3: Optional[float] = None
    gt_tnm_stage: Optional[str] = None
    ai_tnm_stage: Optional[str] = None

class SlicePredictionResponse(BaseModel):
    original_image: str  # Base64 data URL
    predicted_mask: str  # Base64 data URL
    overlay_image: str   # Base64 data URL
    ground_truth_overlay: Optional[str] = None
    kidney_area_cm2: float
    tumor_area_cm2: float
    cyst_area_cm2: float
    pixel_counts: Dict[str, int]
    inference_time_ms: float
    tta_enabled: Optional[bool] = True
    kidney_absence_filter: Optional[Dict[str, int]] = None

class Patient3DMetrics(BaseModel):
    case_id: str
    slices_analyzed: int
    right_kidney_vol_cm3: float
    left_kidney_vol_cm3: float
    total_parenchyma_vol_cm3: float
    tumor_vol_cm3: float
    cyst_vol_cm3: float
    tumor_burden_pct: float
    length_cm: float
    width_cm: float
    thickness_cm: float
    ellipsoid_vol_cm3: float
    ellipsoid_error_pct: float
    tnm_stage: str
    surgical_recommendation: str
    inference_time_sec: float
    gt_comparison: Optional[Dict[str, Any]] = None
    tta_enabled: Optional[bool] = True

class ClinicalReportResponse(BaseModel):
    case_id: str
    markdown_content: str
    structured_json: Dict[str, Any]

class CaseUploadSliceItem(BaseModel):
    slice_index: int
    filename: str
    kidney_area_cm2: float
    tumor_area_cm2: float
    cyst_area_cm2: float
    has_kidney: bool
    has_tumor: bool
    has_cyst: bool

class CaseUploadPreviewSlice(BaseModel):
    slice_index: int
    filename: str
    original_image: str   # base64 data URL
    predicted_mask: str   # base64 data URL
    overlay_image: str    # base64 data URL
    kidney_area_cm2: float
    tumor_area_cm2: float
    cyst_area_cm2: float

class CaseUploadResponse(BaseModel):
    case_id: str
    slices_analyzed: int
    best_preview_slice: int
    right_kidney_vol_cm3: float
    left_kidney_vol_cm3: float
    total_parenchyma_vol_cm3: float
    tumor_vol_cm3: float
    cyst_vol_cm3: float
    tumor_burden_pct: float
    length_cm: float
    width_cm: float
    thickness_cm: float
    ellipsoid_vol_cm3: float
    ellipsoid_error_pct: float
    tnm_stage: str
    surgical_recommendation: str
    inference_time_sec: float
    tta_enabled: Optional[bool] = True
    kidney_absence_filter: Optional[Dict[str, int]] = None
    slices: List[CaseUploadSliceItem]
    key_preview_slices: Optional[List[CaseUploadPreviewSlice]] = []
    report_markdown: str
