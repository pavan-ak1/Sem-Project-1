import io
import time
import base64
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from scipy.ndimage import label, binary_dilation

from backend.model_loader import load_model, get_device
import os
import re
import json
from backend.clinical_analytics import (
    calculate_physical_volumes,
    compute_pca_renal_axes,
    compute_tnm_surgical_staging,
    generate_clinical_markdown_report,
    VOXEL_SPACING_DEFAULT
)

DATA_ROOT = Path(os.getenv("DATA_ROOT", r"D:\archive"))
IMAGE_DIR = DATA_ROOT / "images"
MASK_DIR  = DATA_ROOT / "masks"
CSV_PATH  = DATA_ROOT / "dataset.csv"
UPLOADED_CASES_DIR = Path(os.getenv("UPLOADED_CASES_DIR", "uploaded_cases")).resolve()
UPLOADED_CASES_DIR.mkdir(parents=True, exist_ok=True)


# Color constants normalized 0-255
PALETTE = {
    0: (0, 0, 0, 0),         # Background: transparent
    1: (16, 185, 129, 140),  # Kidney: Emerald Green (#10B981)
    2: (239, 68, 68, 180),   # Tumor: Crimson Red (#EF4444)
    3: (59, 130, 246, 160),  # Cyst: Cerulean Blue (#3B82F6)
}


def standardize_dims(image: np.ndarray, target_h: int = 512, target_w: int = 512) -> np.ndarray:
    """Center-crops or constant-pads image to (512, 512). Handles 2D (H, W) and multi-channel arrays."""
    # Convert multi-channel (RGB, RGBA) to 2D single-channel grayscale if needed
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
        elif image.shape[2] == 4:
            image = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
        elif image.shape[2] == 1:
            image = image[:, :, 0]
        else:
            image = np.mean(image, axis=2)
    elif image.ndim > 3:
        image = np.squeeze(image)
        if image.ndim > 2:
            image = image[0]

    h, w = image.shape[:2]
    if h == target_h and w == target_w:
        return image

    if h > target_h:
        top = (h - target_h) // 2
        image = image[top:top + target_h, :]
    elif h < target_h:
        top = (target_h - h) // 2
        bottom = target_h - h - top
        image = np.pad(image, ((top, bottom), (0, 0)), mode="constant", constant_values=0)

    if w > target_w:
        left = (w - target_w) // 2
        image = image[:, left:left + target_w]
    elif w < target_w:
        left = (target_w - w) // 2
        right = target_w - w - left
        image = np.pad(image, ((0, 0), (left, right)), mode="constant", constant_values=0)

    return image


def image_to_base64_url(pil_img: Image.Image, format: str = "PNG") -> str:
    buffered = io.BytesIO()
    pil_img.save(buffered, format=format)
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


def create_overlay(image_norm: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Creates RGBA overlay of mask over grayscale CT image."""
    h, w = mask.shape
    base_rgb = (np.clip(image_norm, 0, 1) * 255).astype(np.uint8)
    base_img = Image.fromarray(base_rgb).convert("RGBA")

    overlay_arr = np.zeros((h, w, 4), dtype=np.uint8)
    for c, rgba in PALETTE.items():
        if c == 0:
            continue
        match = (mask == c)
        overlay_arr[match] = rgba

    overlay_img = Image.fromarray(overlay_arr, mode="RGBA")
    combined = Image.alpha_composite(base_img, overlay_img)
    return combined


def clean_segmentation_volume(raw_volume: np.ndarray) -> np.ndarray:
    """3D Connected Component Analysis morphological cleanup with exophytic tumor preservation."""
    clean = np.zeros_like(raw_volume)
    is_single_slice = (raw_volume.shape[0] == 1)

    # 1. Kidney: retain top 2 largest components (left and right kidneys)
    # Adapt minimum voxel thresholds if processing a single 2D slice vs full 3D series
    min_k_vox = 20 if is_single_slice else 100
    min_t_vox = 10 if is_single_slice else 25
    min_c_vox = 5 if is_single_slice else 15

    kidney_voxels = (raw_volume == 1)
    if np.any(kidney_voxels):
        labeled_k, n_k = label(kidney_voxels)
        if n_k > 0:
            sizes = [np.sum(labeled_k == i) for i in range(1, n_k + 1)]
            top_k_indices = np.argsort(sizes)[::-1][:2]
            for idx in top_k_indices:
                if sizes[idx] >= min_k_vox:
                    clean[labeled_k == (idx + 1)] = 1

    # 2. Anatomical containment:
    # Large renal cell carcinomas (RCC) can grow up to 8-10 cm outward (exophytic growth into retroperitoneum).
    # We use a wide envelope (25 iterations ~20 mm) for tumor to eliminate distant bowel/liver noise
    # without truncating large cancer masses, and a tighter envelope for benign intra-renal cysts.
    if np.any(clean == 1):
        kidney_envelope_wide = binary_dilation(clean == 1, iterations=25)
        kidney_envelope_tight = binary_dilation(clean == 1, iterations=15)
    else:
        kidney_envelope_wide = np.ones_like(raw_volume, dtype=bool)
        kidney_envelope_tight = np.ones_like(raw_volume, dtype=bool)

    # 3. Tumor: filtered by wide anatomical boundary and minimum cluster threshold
    tumor_voxels = (raw_volume == 2) & kidney_envelope_wide
    if np.any(tumor_voxels):
        labeled_t, n_t = label(tumor_voxels)
        if n_t > 0:
            sizes_t = [np.sum(labeled_t == i) for i in range(1, n_t + 1)]
            for i, sz in enumerate(sizes_t, 1):
                if sz >= min_t_vox:
                    clean[labeled_t == i] = 2

    # 4. Cyst: benign fluid-attenuated cysts remain intra-parenchymal or subcapsular
    cyst_voxels = (raw_volume == 3) & kidney_envelope_tight
    if np.any(cyst_voxels):
        labeled_c, n_c = label(cyst_voxels)
        if n_c > 0:
            sizes_c = [np.sum(labeled_c == i) for i in range(1, n_c + 1)]
            for i, sz in enumerate(sizes_c, 1):
                if sz >= min_c_vox:
                    clean[labeled_c == i] = 3

    return clean


def predict_tensor(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
    use_tta: bool = False
) -> np.ndarray:
    """
    Executes model forward pass with optional 4-Fold Test-Time Augmentation (TTA).
    tensor: (B, 1, 512, 512) float tensor in [0.0, 1.0].
    Returns: (B, 512, 512) uint8 numpy array of class indices {0, 1, 2, 3}.
    """
    with torch.inference_mode():
        if not use_tta:
            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(tensor)
            else:
                logits = model(tensor)
            preds = torch.argmax(logits, dim=1)
        else:
            # 4-Fold Multi-Axis TTA: original, horizontal flip, vertical flip, and diagonal flip
            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    p1 = F.softmax(model(tensor), dim=1)
                    p2 = torch.flip(F.softmax(model(torch.flip(tensor, dims=[3])), dim=1), dims=[3])
                    p3 = torch.flip(F.softmax(model(torch.flip(tensor, dims=[2])), dim=1), dims=[2])
                    p4 = torch.flip(F.softmax(model(torch.flip(tensor, dims=[2, 3])), dim=1), dims=[2, 3])
            else:
                p1 = F.softmax(model(tensor), dim=1)
                p2 = torch.flip(F.softmax(model(torch.flip(tensor, dims=[3])), dim=1), dims=[3])
                p3 = torch.flip(F.softmax(model(torch.flip(tensor, dims=[2])), dim=1), dims=[2])
                p4 = torch.flip(F.softmax(model(torch.flip(tensor, dims=[2, 3])), dim=1), dims=[2, 3])

            probs_avg = 0.25 * (p1 + p2 + p3 + p4)
            preds = torch.argmax(probs_avg, dim=1)

    return preds.cpu().numpy().astype(np.uint8)


def run_single_slice_inference(image_bytes: bytes, use_tta: Optional[bool] = None) -> Dict[str, Any]:
    """Inference on a single uploaded slice with 16-bit/8-bit auto-detection and adaptive TTA."""
    model, device = load_model()
    if use_tta is None:
        use_tta = (device.type == "cuda")
    t_start = time.perf_counter()

    pil_raw = Image.open(io.BytesIO(image_bytes))
    if pil_raw.mode in ("RGB", "RGBA", "P"):
        pil_raw = pil_raw.convert("L")
    img_arr = np.array(pil_raw, dtype=np.float32)

    # Standardization & Normalization
    img_std = standardize_dims(img_arr, 512, 512)
    is_16bit = (img_std.max() > 255.0) or (pil_raw.mode in ["I;16", "I", "F"])

    if is_16bit:
        norm_img = img_std / 65535.0
    else:
        norm_img = (img_std - img_std.min()) / max(1e-4, img_std.max() - img_std.min())

    tensor_in = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).to(device)
    pred_mask = predict_tensor(model, tensor_in, device, use_tta=use_tta)[0]

    t_elapsed = (time.perf_counter() - t_start) * 1000.0

    # Physical 2D area (pixel spacing 0.78125 mm * 0.78125 mm = 0.0061035 cm2)
    px_area_cm2 = (0.78125 * 0.78125) / 100.0
    px_k = int(np.sum(pred_mask == 1))
    px_t = int(np.sum(pred_mask == 2))
    px_c = int(np.sum(pred_mask == 3))

    # Base64 representations
    vis_norm = np.clip(norm_img, 0.0, 1.0)
    orig_b64 = image_to_base64_url(Image.fromarray((vis_norm * 255).astype(np.uint8)))
    mask_b64 = image_to_base64_url(Image.fromarray(pred_mask * 80).convert("L"))
    overlay_img = create_overlay(vis_norm, pred_mask)
    overlay_b64 = image_to_base64_url(overlay_img)

    return {
        "original_image": orig_b64,
        "predicted_mask": mask_b64,
        "overlay_image": overlay_b64,
        "ground_truth_overlay": None,
        "kidney_area_cm2": round(px_k * px_area_cm2, 2),
        "tumor_area_cm2": round(px_t * px_area_cm2, 2),
        "cyst_area_cm2": round(px_c * px_area_cm2, 2),
        "pixel_counts": {
            "kidney": px_k,
            "tumor": px_t,
            "cyst": px_c,
            "background": int(np.sum(pred_mask == 0))
        },
        "inference_time_ms": round(t_elapsed, 1),
        "tta_enabled": use_tta
    }


def run_patient_case_inference(case_id: str, use_tta: Optional[bool] = None) -> Dict[str, Any]:
    """Runs 3D volume reconstruction and clinical decision support for a patient case with adaptive TTA."""
    model, device = load_model()
    if use_tta is None:
        use_tta = (device.type == "cuda")
    t_start = time.perf_counter()

    if not CSV_PATH.exists():
        raise FileNotFoundError("dataset.csv not found!")

    df = pd.read_csv(CSV_PATH)
    case_df = df[df["case_id"] == case_id].sort_values("slice_index").reset_index(drop=True)
    if len(case_df) == 0:
        raise ValueError(f"Case ID '{case_id}' not found in dataset!")

    slices_list = []
    gt_list = []
    has_gt = False

    for _, row in case_df.iterrows():
        img_p = IMAGE_DIR / row["filename"]
        if not img_p.exists():
            continue
        p_raw = Image.open(img_p)
        arr = np.array(p_raw, dtype=np.float32)
        arr = standardize_dims(arr, 512, 512)
        if arr.max() > 255.0 or p_raw.mode in ["I;16", "I", "F"]:
            norm = arr / 65535.0
        else:
            norm = (arr - arr.min()) / max(1e-4, arr.max() - arr.min())
        slices_list.append(norm)

        mask_p = MASK_DIR / row["filename"]
        if mask_p.exists():
            has_gt = True
            m_arr = np.array(Image.open(mask_p), dtype=np.uint8)
            m_arr = standardize_dims(m_arr, 512, 512)
            gt_list.append(m_arr)

    if len(slices_list) == 0:
        raise ValueError(f"No image files found for case {case_id}")

    # Batch model forward pass with adaptive TTA and cache-optimized batch size
    vol_3d_in = np.stack(slices_list, axis=0)  # (D, 512, 512)
    preds_3d = []

    batch_size = 16 if device.type == "cuda" else 4
    with torch.inference_mode():
        for i in range(0, len(vol_3d_in), batch_size):
            batch_np = vol_3d_in[i:i + batch_size]
            b_tensor = torch.from_numpy(batch_np).unsqueeze(1).to(device)
            p_batch = predict_tensor(model, b_tensor, device, use_tta=use_tta)
            preds_3d.append(p_batch)

    raw_preds_3d = np.concatenate(preds_3d, axis=0)
    clean_preds_3d = clean_segmentation_volume(raw_preds_3d)

    # Clinical volumetric calculations
    vols = calculate_physical_volumes(clean_preds_3d, VOXEL_SPACING_DEFAULT)
    pca = compute_pca_renal_axes(clean_preds_3d == 1, VOXEL_SPACING_DEFAULT)
    staging = compute_tnm_surgical_staging(clean_preds_3d, vols, VOXEL_SPACING_DEFAULT)

    t_elapsed = time.perf_counter() - t_start

    # Ground truth comparison if available
    gt_comparison = None
    if has_gt and len(gt_list) == len(slices_list):
        gt_3d = np.stack(gt_list, axis=0)
        gt_vols = calculate_physical_volumes(gt_3d, VOXEL_SPACING_DEFAULT)
        gt_pca = compute_pca_renal_axes(gt_3d == 1, VOXEL_SPACING_DEFAULT)
        gt_staging = compute_tnm_surgical_staging(gt_3d, gt_vols, VOXEL_SPACING_DEFAULT)

        # 3D Dice calculation
        def dice_3d(pred, gt, cls_idx):
            p = (pred == cls_idx)
            g = (gt == cls_idx)
            intersection = np.logical_and(p, g).sum()
            total = p.sum() + g.sum()
            return float((2.0 * intersection) / total) if total > 0 else 1.0

        gt_comparison = {
            "gt_total_kidney_vol_cm3": gt_vols["total_parenchyma_vol_cm3"],
            "gt_tumor_vol_cm3": gt_vols["tumor_vol_cm3"],
            "gt_tnm_stage": gt_staging["tnm_t_stage"],
            "gt_d_max_cm": gt_staging["d_max_cm"],
            "dice_kidney": round(dice_3d(clean_preds_3d, gt_3d, 1), 4),
            "dice_tumor": round(dice_3d(clean_preds_3d, gt_3d, 2), 4),
            "dice_cyst": round(dice_3d(clean_preds_3d, gt_3d, 3), 4),
        }

    report_md = generate_clinical_markdown_report(case_id, vols, pca, staging, t_elapsed)

    return {
        "case_id": case_id,
        "slices_analyzed": len(slices_list),
        "right_kidney_vol_cm3": vols["right_kidney_vol_cm3"],
        "left_kidney_vol_cm3": vols["left_kidney_vol_cm3"],
        "total_parenchyma_vol_cm3": vols["total_parenchyma_vol_cm3"],
        "tumor_vol_cm3": vols["tumor_vol_cm3"],
        "cyst_vol_cm3": vols["cyst_vol_cm3"],
        "tumor_burden_pct": vols["tumor_burden_pct"],
        "length_cm": pca["length_cm"],
        "width_cm": pca["width_cm"],
        "thickness_cm": pca["thickness_cm"],
        "ellipsoid_vol_cm3": pca["ellipsoid_vol_cm3"],
        "ellipsoid_error_pct": pca["ellipsoid_error_pct"],
        "tnm_stage": staging["tnm_t_stage"],
        "surgical_recommendation": staging["surgical_recommendation"],
        "inference_time_sec": round(t_elapsed, 2),
        "gt_comparison": gt_comparison,
        "report_markdown": report_md,
        "tta_enabled": use_tta
    }


def natural_sort_key(s: str):
    """Natural sorting key to preserve correct anatomical slice z-ordering."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def run_uploaded_case_pipeline(
    files: List[Tuple[str, bytes]],
    case_name: Optional[str] = None,
    use_tta: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Executes full 3D volumetric analysis, CCA morphological cleaning,
    PCA caliper morphometry, AJCC TNM clinical staging, and clinical report
    generation for an uploaded patient case folder or multi-slice series with adaptive TTA.
    """
    model, device = load_model()
    if use_tta is None:
        use_tta = (device.type == "cuda")
    t_start = time.perf_counter()

    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dcm"}
    filtered_files = [f for f in files if Path(f[0]).suffix.lower() in valid_exts or not Path(f[0]).suffix]
    if not filtered_files:
        filtered_files = files

    # Natural sorting by filename so slices are in correct axial z-order
    sorted_files = sorted(filtered_files, key=lambda f: natural_sort_key(Path(f[0]).name))

    # Derive case ID if not supplied
    if not case_name or not case_name.strip():
        first_fname = Path(sorted_files[0][0]).name
        match = re.search(r'(case_\d+)', first_fname, re.IGNORECASE)
        if match:
            case_id = match.group(1).lower()
        else:
            parent_dir = Path(sorted_files[0][0]).parts
            if len(parent_dir) > 1 and parent_dir[0]:
                case_id = parent_dir[0]
            else:
                case_id = f"uploaded_case_{int(time.time())}"
    else:
        case_id = case_name.strip()

    # Load and standardize images
    slices_list = []
    vis_images_8bit = []
    filenames = []

    for fname, raw_bytes in sorted_files:
        try:
            p_raw = Image.open(io.BytesIO(raw_bytes))
            if p_raw.mode in ("RGB", "RGBA", "P"):
                p_raw = p_raw.convert("L")
            arr = np.array(p_raw, dtype=np.float32)
            std_arr = standardize_dims(arr, 512, 512)

            is_16bit = (std_arr.max() > 255.0) or (p_raw.mode in ["I;16", "I", "F"])
            if is_16bit:
                norm = std_arr / 65535.0
            else:
                norm = (std_arr - std_arr.min()) / max(1e-4, std_arr.max() - std_arr.min())

            slices_list.append(norm)
            vis_8bit = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
            vis_images_8bit.append(vis_8bit)
            filenames.append(Path(fname).name)
        except Exception as e:
            print(f"Skipping corrupt slice file {fname}: {e}")
            continue

    if len(slices_list) == 0:
        raise ValueError("No valid image files found in the uploaded case folder.")

    # Batch model forward pass with adaptive TTA and cache-optimized batch size
    vol_3d_in = np.stack(slices_list, axis=0)  # (D, 512, 512)
    preds_3d = []
    batch_size = 16 if device.type == "cuda" else 4

    with torch.inference_mode():
        for i in range(0, len(vol_3d_in), batch_size):
            batch_np = vol_3d_in[i:i + batch_size]
            b_tensor = torch.from_numpy(batch_np).unsqueeze(1).to(device)
            p_batch = predict_tensor(model, b_tensor, device, use_tta=use_tta)
            preds_3d.append(p_batch)

    raw_preds_3d = np.concatenate(preds_3d, axis=0)
    clean_preds_3d = clean_segmentation_volume(raw_preds_3d)

    # Physical volume, PCA calipers, TNM staging
    vols = calculate_physical_volumes(clean_preds_3d, VOXEL_SPACING_DEFAULT)
    pca = compute_pca_renal_axes(clean_preds_3d == 1, VOXEL_SPACING_DEFAULT)
    staging = compute_tnm_surgical_staging(clean_preds_3d, vols, VOXEL_SPACING_DEFAULT)

    t_elapsed = time.perf_counter() - t_start
    report_md = generate_clinical_markdown_report(case_id, vols, pca, staging, t_elapsed)

    # Save to uploaded_cases directory cache for on-demand slice viewer
    case_store_dir = UPLOADED_CASES_DIR / case_id
    case_images_dir = case_store_dir / "images"
    case_masks_dir = case_store_dir / "masks"
    case_images_dir.mkdir(parents=True, exist_ok=True)
    case_masks_dir.mkdir(parents=True, exist_ok=True)

    px_area_cm2 = (0.78125 * 0.78125) / 100.0
    slices_summary = []
    tumor_slice_indices = []
    kidney_slice_indices = []

    for i in range(len(slices_list)):
        k_px = int(np.sum(clean_preds_3d[i] == 1))
        t_px = int(np.sum(clean_preds_3d[i] == 2))
        c_px = int(np.sum(clean_preds_3d[i] == 3))

        if t_px > 0:
            tumor_slice_indices.append((i, t_px))
        if k_px > 0:
            kidney_slice_indices.append((i, k_px))

        # Save slice and mask on disk for on-demand inspection
        slice_img_path = case_images_dir / f"slice_{i:04d}.png"
        mask_img_path = case_masks_dir / f"mask_{i:04d}.png"
        Image.fromarray(vis_images_8bit[i]).save(slice_img_path)
        Image.fromarray(clean_preds_3d[i]).save(mask_img_path)

        slices_summary.append({
            "slice_index": i,
            "filename": filenames[i],
            "kidney_area_cm2": round(k_px * px_area_cm2, 2),
            "tumor_area_cm2": round(t_px * px_area_cm2, 2),
            "cyst_area_cm2": round(c_px * px_area_cm2, 2),
            "has_kidney": k_px > 0,
            "has_tumor": t_px > 0,
            "has_cyst": c_px > 0
        })

    # Determine best preview slice
    if tumor_slice_indices:
        best_preview_slice = max(tumor_slice_indices, key=lambda x: x[1])[0]
    elif kidney_slice_indices:
        best_preview_slice = max(kidney_slice_indices, key=lambda x: x[1])[0]
    else:
        best_preview_slice = max(0, min(len(slices_list) - 1, len(slices_list) // 2))

    # Generate key preview slices with base64 for instant display
    candidate_preview_indices = []
    if tumor_slice_indices:
        candidate_preview_indices.append(best_preview_slice)
    if kidney_slice_indices and kidney_slice_indices[0][0] not in candidate_preview_indices:
        candidate_preview_indices.append(kidney_slice_indices[0][0])
    mid_idx = max(0, min(len(slices_list) - 1, len(slices_list) // 2))
    if mid_idx not in candidate_preview_indices:
        candidate_preview_indices.append(mid_idx)

    if not candidate_preview_indices:
        candidate_preview_indices.append(0)

    candidate_preview_indices = sorted(list(set(candidate_preview_indices)))[:4]
    key_previews = []
    for p_idx in candidate_preview_indices:
        norm_sl = slices_list[p_idx]
        m_sl = clean_preds_3d[p_idx]
        orig_b64 = image_to_base64_url(Image.fromarray(vis_images_8bit[p_idx]))
        mask_b64 = image_to_base64_url(Image.fromarray(m_sl * 80).convert("L"))
        overlay_b64 = image_to_base64_url(create_overlay(norm_sl, m_sl))
        key_previews.append({
            "slice_index": p_idx,
            "filename": filenames[p_idx],
            "original_image": orig_b64,
            "predicted_mask": mask_b64,
            "overlay_image": overlay_b64,
            "kidney_area_cm2": slices_summary[p_idx]["kidney_area_cm2"],
            "tumor_area_cm2": slices_summary[p_idx]["tumor_area_cm2"],
            "cyst_area_cm2": slices_summary[p_idx]["cyst_area_cm2"]
        })

    # Save meta.json
    meta_data = {
        "case_id": case_id,
        "slices_analyzed": len(slices_list),
        "best_preview_slice": best_preview_slice,
        "vols": vols,
        "pca": pca,
        "staging": staging,
        "inference_time_sec": round(t_elapsed, 2),
        "report_markdown": report_md,
        "tta_enabled": use_tta,
        "slices": slices_summary
    }
    with open(case_store_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)

    return {
        "case_id": case_id,
        "slices_analyzed": len(slices_list),
        "best_preview_slice": best_preview_slice,
        "right_kidney_vol_cm3": vols["right_kidney_vol_cm3"],
        "left_kidney_vol_cm3": vols["left_kidney_vol_cm3"],
        "total_parenchyma_vol_cm3": vols["total_parenchyma_vol_cm3"],
        "tumor_vol_cm3": vols["tumor_vol_cm3"],
        "cyst_vol_cm3": vols["cyst_vol_cm3"],
        "tumor_burden_pct": vols["tumor_burden_pct"],
        "length_cm": pca["length_cm"],
        "width_cm": pca["width_cm"],
        "thickness_cm": pca["thickness_cm"],
        "ellipsoid_vol_cm3": pca["ellipsoid_vol_cm3"],
        "ellipsoid_error_pct": pca["ellipsoid_error_pct"],
        "tnm_stage": staging["tnm_t_stage"],
        "surgical_recommendation": staging["surgical_recommendation"],
        "inference_time_sec": round(t_elapsed, 2),
        "tta_enabled": use_tta,
        "slices": slices_summary,
        "key_preview_slices": key_previews,
        "report_markdown": report_md
    }


def get_uploaded_case_slice(case_id: str, slice_idx: int) -> Dict[str, Any]:
    """Retrieves on-demand slice visualization for an uploaded case."""
    case_store_dir = UPLOADED_CASES_DIR / case_id
    if not case_store_dir.exists():
        raise FileNotFoundError(f"Uploaded case '{case_id}' not found.")

    slice_img_path = case_store_dir / "images" / f"slice_{slice_idx:04d}.png"
    mask_img_path = case_store_dir / "masks" / f"mask_{slice_idx:04d}.png"
    if not slice_img_path.exists() or not mask_img_path.exists():
        raise FileNotFoundError(f"Slice {slice_idx} not found for uploaded case '{case_id}'.")

    raw_img = Image.open(slice_img_path).convert("L")
    mask_img = Image.open(mask_img_path)

    img_arr = np.array(raw_img, dtype=np.float32)
    norm = img_arr / 255.0
    mask_arr = np.array(mask_img, dtype=np.uint8)

    px_area_cm2 = (0.78125 * 0.78125) / 100.0
    k_px = int(np.sum(mask_arr == 1))
    t_px = int(np.sum(mask_arr == 2))
    c_px = int(np.sum(mask_arr == 3))

    orig_b64 = image_to_base64_url(raw_img)
    mask_b64 = image_to_base64_url(Image.fromarray(mask_arr * 80).convert("L"))
    overlay_img = create_overlay(norm, mask_arr)
    overlay_b64 = image_to_base64_url(overlay_img)

    return {
        "case_id": case_id,
        "slice_index": slice_idx,
        "original_image": orig_b64,
        "predicted_mask": mask_b64,
        "overlay_image": overlay_b64,
        "kidney_area_cm2": round(k_px * px_area_cm2, 2),
        "tumor_area_cm2": round(t_px * px_area_cm2, 2),
        "cyst_area_cm2": round(c_px * px_area_cm2, 2),
        "pixel_counts": {
            "kidney": k_px,
            "tumor": t_px,
            "cyst": c_px,
            "background": int(np.sum(mask_arr == 0))
        }
    }
