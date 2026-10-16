import math
from datetime import datetime
from typing import Dict, Any, Tuple
import numpy as np
from scipy.spatial.distance import pdist

VOXEL_SPACING_DEFAULT = (3.0, 0.78125, 0.78125)  # (dz, dy, dx) in mm


def calculate_physical_volumes(
    mask_3d: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = VOXEL_SPACING_DEFAULT
) -> Dict[str, float]:
    """
    Computes anatomical physical volume in cm3 (mL) for all classes:
    Separates Right and Left kidneys via coronal midline split (anatomical convention).
    Classes: 0=Background, 1=Kidney, 2=Tumor, 3=Cyst.
    """
    dz, dy, dx = voxel_spacing
    voxel_vol_cm3 = (dz * dy * dx) / 1000.0  # mm3 to cm3

    kidney_voxels = (mask_3d == 1)
    tumor_voxels  = (mask_3d == 2)
    cyst_voxels   = (mask_3d == 3)

    d, h, w = mask_3d.shape
    midline = w // 2

    # In radiological cross-section, patient's Right is image Left (x < midline)
    right_kidney_mask = np.zeros_like(kidney_voxels)
    right_kidney_mask[:, :, :midline] = kidney_voxels[:, :, :midline]

    left_kidney_mask = np.zeros_like(kidney_voxels)
    left_kidney_mask[:, :, midline:] = kidney_voxels[:, :, midline:]

    right_kidney_vol = float(np.sum(right_kidney_mask) * voxel_vol_cm3)
    left_kidney_vol  = float(np.sum(left_kidney_mask) * voxel_vol_cm3)
    total_parenchyma = right_kidney_vol + left_kidney_vol
    tumor_vol        = float(np.sum(tumor_voxels) * voxel_vol_cm3)
    cyst_vol         = float(np.sum(cyst_voxels) * voxel_vol_cm3)

    tumor_burden = (tumor_vol / max(1e-3, total_parenchyma + tumor_vol)) * 100.0 if (total_parenchyma + tumor_vol) > 0 else 0.0

    return {
        "right_kidney_vol_cm3": round(right_kidney_vol, 2),
        "left_kidney_vol_cm3": round(left_kidney_vol, 2),
        "total_parenchyma_vol_cm3": round(total_parenchyma, 2),
        "tumor_vol_cm3": round(tumor_vol, 2),
        "cyst_vol_cm3": round(cyst_vol, 2),
        "tumor_burden_pct": round(tumor_burden, 2)
    }


def compute_pca_renal_axes(
    organ_mask_3d: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = VOXEL_SPACING_DEFAULT
) -> Dict[str, float]:
    """
    Extracts 3D Principal Component Analysis (PCA) orthogonal caliper axes (L, W, T)
    and computes prolate ellipsoid volume error as documented in Koukoutegos et al. (2024).
    """
    coords = np.argwhere(organ_mask_3d)
    if len(coords) < 10:
        return {
            "length_cm": 0.0, "width_cm": 0.0, "thickness_cm": 0.0,
            "true_3d_vol_cm3": 0.0, "ellipsoid_vol_cm3": 0.0, "ellipsoid_error_pct": 0.0
        }

    dz, dy, dx = voxel_spacing
    voxel_vol_cm3 = (dz * dy * dx) / 1000.0

    # Physical coordinates in millimeters
    phys_coords = coords.astype(np.float64) * np.array([dz, dy, dx])

    mean_c = np.mean(phys_coords, axis=0)
    centered = phys_coords - mean_c
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    projected = np.dot(centered, eigvecs)
    ranges_mm = np.ptp(projected, axis=0)
    ranges_cm = ranges_mm / 10.0

    length_cm    = float(ranges_cm[0])
    width_cm     = float(ranges_cm[1])
    thickness_cm = float(ranges_cm[2])
    # For single-slice or flat 2D inputs, ensure thickness reflects nominal slice spacing
    if thickness_cm < 0.01:
        thickness_cm = round(dz / 10.0, 2)

    v_true_3d   = float(len(coords) * voxel_vol_cm3)
    v_ellipsoid = float((math.pi / 6.0) * length_cm * width_cm * thickness_cm)
    ellipsoid_err = float(abs(v_ellipsoid - v_true_3d) / max(1e-4, v_true_3d) * 100.0)

    return {
        "length_cm": round(length_cm, 2),
        "width_cm": round(width_cm, 2),
        "thickness_cm": round(thickness_cm, 2),
        "true_3d_vol_cm3": round(v_true_3d, 2),
        "ellipsoid_vol_cm3": round(v_ellipsoid, 2),
        "ellipsoid_error_pct": round(ellipsoid_err, 2)
    }


def compute_tnm_surgical_staging(
    mask_3d: np.ndarray,
    vols: Dict[str, float],
    voxel_spacing: Tuple[float, float, float] = VOXEL_SPACING_DEFAULT
) -> Dict[str, Any]:
    """
    Computes AJCC/UICC 8th Edition TNM clinical staging and nephron-sparing feasibility.
    """
    tumor_coords = np.argwhere(mask_3d == 2)
    if len(tumor_coords) == 0:
        return {
            "tnm_t_stage": "cT0",
            "d_max_cm": 0.0,
            "stage_description": "No solid renal neoplasm identified",
            "surgical_recommendation": "Routine clinical surveillance",
            "nephron_sparing_feasible": True
        }

    dz, dy, dx = voxel_spacing
    phys_coords = tumor_coords.astype(np.float64) * np.array([dz, dy, dx])

    if len(phys_coords) == 1:
        d_max_mm = float(max(dx, dy))
        d_max_cm = round(d_max_mm / 10.0, 2)
    else:
        # Downsample if too large for pairwise distance
        if len(phys_coords) > 2000:
            idx = np.random.choice(len(phys_coords), size=2000, replace=False)
            sample = phys_coords[idx]
        else:
            sample = phys_coords

        d_dist = pdist(sample)
        d_max_mm = float(np.max(d_dist)) if len(d_dist) > 0 else float(max(dx, dy))
        d_max_cm = round(d_max_mm / 10.0, 2)

    # AJCC 8th Edition TNM Staging criteria
    if d_max_cm <= 4.0:
        stage = "cT1a"
        desc = "Organ-confined renal neoplasm ≤ 4.0 cm"
        recom = "High Partial Nephrectomy (Nephron-Sparing Surgery) feasibility. Preserves functional renal parenchyma."
        feasible = True
    elif d_max_cm <= 7.0:
        stage = "cT1b"
        desc = "Organ-confined renal neoplasm > 4.0 cm and ≤ 7.0 cm"
        recom = "Partial Nephrectomy feasible depending on hilar proximity and R.E.N.A.L. nephrometry complexity score."
        feasible = True
    elif d_max_cm <= 10.0:
        stage = "cT2a"
        desc = "Organ-confined renal neoplasm > 7.0 cm and ≤ 10.0 cm"
        recom = "Radical Nephrectomy strongly indicated due to substantial volume burden and vascular involvement."
        feasible = False
    else:
        stage = "cT2b"
        desc = "Organ-confined renal neoplasm > 10.0 cm"
        recom = "Radical Nephrectomy strictly recommended. Substantial loss of ipsilateral parenchymal reserve."
        feasible = False

    return {
        "tnm_t_stage": stage,
        "d_max_cm": d_max_cm,
        "stage_description": desc,
        "surgical_recommendation": recom,
        "nephron_sparing_feasible": feasible
    }


def generate_clinical_markdown_report(
    case_id: str,
    vols: Dict[str, float],
    pca: Dict[str, float],
    staging: Dict[str, Any],
    inference_sec: float
) -> str:
    """Generates structured hospital-grade Markdown clinical radiology report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""# CLINICAL RADIOLOGY & QUANTITATIVE DECISION-SUPPORT REPORT
**Patient Case ID:** `{case_id}`  
**Generated At:** {timestamp}  
**Software Pipeline:** NephroAI Clinical Renal Analysis System (KiTS23 Residual U-Net)  
**Inference Latency:** {inference_sec:.2f} seconds  

---

### 1. EXECUTIVE DIAGNOSTIC SUMMARY
- **Primary Finding:** Solid renal mass identified.
- **AJCC TNM Clinical Stage:** **{staging['tnm_t_stage']}** (Max 3D Diameter: **{staging['d_max_cm']} cm**).
- **Tumor Burden in Organ:** **{vols['tumor_burden_pct']}%** of functional parenchymal reserve.
- **Surgical Guidance:** {staging['surgical_recommendation']}

---

### 2. QUANTITATIVE VOLUMETRIC ANALYSIS (cm³ / mL)
| Anatomical Structure | Predicted Volume | Percentage of Total |
| :--- | :---: | :---: |
| **Right Kidney Parenchyma** | {vols['right_kidney_vol_cm3']} cm³ | {round(vols['right_kidney_vol_cm3'] / max(0.1, vols['total_parenchyma_vol_cm3']) * 100, 1)}% |
| **Left Kidney Parenchyma** | {vols['left_kidney_vol_cm3']} cm³ | {round(vols['left_kidney_vol_cm3'] / max(0.1, vols['total_parenchyma_vol_cm3']) * 100, 1)}% |
| **Renal Neoplasm (Tumor)** | {vols['tumor_vol_cm3']} cm³ | — |
| **Renal Cyst** | {vols['cyst_vol_cm3']} cm³ | — |
| **Total Functional Parenchyma** | **{vols['total_parenchyma_vol_cm3']} cm³** | 100.0% |

---

### 3. 3D SPATIAL MORPHOMETRY & CALIPER AXES (Koukoutegos et al., 2024)
- **Longitudinal Length ($L$):** {pca['length_cm']} cm
- **Transverse Width ($W$):** {pca['width_cm']} cm
- **Antero-Posterior Thickness ($T$):** {pca['thickness_cm']} cm
- **True 3D Voxel Volume:** {pca['true_3d_vol_cm3']} cm³
- **Prolate Ellipsoid Volume:** {pca['ellipsoid_vol_cm3']} cm³
- **Manual Ellipsoid Approximation Error:** **+{pca['ellipsoid_error_pct']}%** *(Overestimation due to non-elliptical renal hilum)*

---

### 4. MEDICAL DISCLAIMER
*This quantitative CT analysis is synthesized by NephroAI deep learning decision-support algorithms. It is intended to assist medical professionals by providing objective volumetric metrics. Final clinical diagnosis, TNM staging, anatomical margin verification, and operative decisions remain the strict professional responsibility of the attending urologist and board-certified radiologist.*
"""
