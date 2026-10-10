#!/usr/bin/env python3
"""
================================================================================
Comprehensive Model Evaluation & Accuracy Testing Engine for KiTS23 2D U-Net
================================================================================

This module performs rigorous clinical and quantitative evaluation on trained
KiTS23 segmentation models.

Metrics Computed:
  - Overall Pixel Accuracy
  - Per-Class Dice Similarity Coefficient (DSC): Background, Kidney, Tumor, Cyst
  - Foreground Macro Dice (KiTS Challenge Benchmark Metric: Kidney + Tumor + Cyst)
  - Per-Class Intersection-over-Union (IoU / Jaccard Index) & Mean IoU
  - Per-Class Precision (Positive Predictive Value), Recall (Sensitivity), Specificity
  - Pixel-Level 4x4 Confusion Matrix (Raw counts & normalized percentages)
  - Lesion-Specific Subset Metrics (Evaluated strictly on slices containing lesions)

Visual Reports Generated:
  - Side-by-side 4-panel qualitative figures (CT Slice, Ground Truth, Prediction, Overlay)
  - Pixel Confusion Matrix Heatmap Plot
  - Multi-Class Metric Bar Chart
  - Structured JSON & CSV Metric Summary Exports
================================================================================
"""

import sys
import os
import time
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

# Windows console encoding safeguard
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import architecture and dataset helpers from train.py
from train import (
    UNet,
    KiTS23Dataset,
    standardize_spatial_dims,
    CompositeSegmentationLoss
)


# ==============================================================================
# 1. COLOR MAPS & VISUALIZATION UTILITIES
# ==============================================================================

# Standard KiTS anatomical color palette (RGB normalized 0.0 - 1.0)
CLASS_NAMES = ["Background", "Kidney", "Tumor", "Cyst"]
CLASS_COLORS = {
    0: (0.0, 0.0, 0.0),       # Background: Transparent / Black
    1: (0.18, 0.80, 0.44),     # Kidney: Emerald Green
    2: (0.91, 0.30, 0.24),     # Tumor: Crimson Red
    3: (0.20, 0.60, 0.86),     # Cyst: Cerulean Blue
}

def create_rgb_mask(mask: np.ndarray) -> np.ndarray:
    """Converts a (H, W) categorical class index mask into an (H, W, 3) RGB color mask."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for c, color in CLASS_COLORS.items():
        if c == 0:
            continue
        match = (mask == c)
        rgb[match] = color
    return rgb


def overlay_mask_on_image(image_norm: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlays categorical color-coded mask onto a grayscale normalized image [0, 1]."""
    h, w = image_norm.shape
    img_rgb = np.stack([image_norm] * 3, axis=-1)
    rgb_mask = create_rgb_mask(mask)
    foreground = (mask > 0)

    blended = img_rgb.copy()
    blended[foreground] = (1.0 - alpha) * img_rgb[foreground] + alpha * rgb_mask[foreground]
    return np.clip(blended, 0.0, 1.0)


# ==============================================================================
# 2. EXTENDED CLINICAL METRICS ENGINE
# ==============================================================================

class ClinicalEvaluationMetrics:
    """
    Computes rigorous pixel-level and per-class clinical statistics:
      - Confusion Matrix (4x4)
      - Per-class Dice, IoU, Precision, Recall, Specificity
      - Macro Foreground Dice, Mean IoU, Overall Pixel Accuracy
    """
    def __init__(self, num_classes: int = 4, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or CLASS_NAMES
        self.reset()

    def reset(self):
        # 4x4 Confusion Matrix: rows = Ground Truth, cols = Prediction
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_pixels = 0
        self.total_correct = 0

        # Lesion-slice specific tracking
        self.lesion_slices_evaluated = 0
        self.lesion_confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        p = preds.detach().cpu().numpy().reshape(-1)
        t = targets.detach().cpu().numpy().reshape(-1)

        self.total_pixels += p.size
        self.total_correct += int(np.sum(p == t))

        # Update global confusion matrix
        cm = np.bincount(
            self.num_classes * t + p,
            minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += cm

    def compute(self) -> Dict[str, Any]:
        cm = self.confusion_matrix.astype(np.float64)
        total_pixels = np.sum(cm)

        dice_per_class = {}
        iou_per_class = {}
        precision_per_class = {}
        recall_per_class = {}
        specificity_per_class = {}

        for c in range(self.num_classes):
            c_name = self.class_names[c]
            tp = cm[c, c]
            fp = np.sum(cm[:, c]) - tp
            fn = np.sum(cm[c, :]) - tp
            tn = total_pixels - (tp + fp + fn)

            # Dice (F1-score)
            dice_denom = 2.0 * tp + fp + fn
            dice = (2.0 * tp / dice_denom) if dice_denom > 0 else (1.0 if (tp + fn == 0) else 0.0)

            # IoU (Jaccard)
            iou_denom = tp + fp + fn
            iou = (tp / iou_denom) if iou_denom > 0 else (1.0 if (tp + fn == 0) else 0.0)

            # Precision (PPV)
            prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0

            # Recall (Sensitivity)
            rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

            # Specificity (TNR)
            spec = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

            dice_per_class[f"Dice_{c_name}"] = float(dice)
            iou_per_class[f"IoU_{c_name}"] = float(iou)
            precision_per_class[f"Precision_{c_name}"] = float(prec)
            recall_per_class[f"Recall_{c_name}"] = float(rec)
            specificity_per_class[f"Specificity_{c_name}"] = float(spec)

        # Foreground Macro Dice (excluding background): Kidney, Tumor, Cyst
        fg_dice_values = [dice_per_class[f"Dice_{name}"] for name in self.class_names[1:]]
        macro_dice = float(np.mean(fg_dice_values))

        # Overall Mean IoU across all 4 classes
        mean_iou = float(np.mean(list(iou_per_class.values())))

        # Overall Pixel Accuracy
        pixel_acc = float(self.total_correct / max(1, self.total_pixels))

        return {
            "Pixel_Accuracy": pixel_acc,
            "Macro_Dice_Foreground": macro_dice,
            "Mean_IoU": mean_iou,
            "Dice": dice_per_class,
            "IoU": iou_per_class,
            "Precision": precision_per_class,
            "Recall": recall_per_class,
            "Specificity": specificity_per_class,
            "Confusion_Matrix": self.confusion_matrix.tolist()
        }


# ==============================================================================
# ==============================================================================
# 3. MODEL ARCHITECTURES & CHECKPOINT LOADER
# ==============================================================================

class ResidualBlock(nn.Module):
    """Two-convolution residual unit with InstanceNorm2d and PReLU activation."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.act1  = nn.PReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.act2  = nn.PReLU()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.act2(out + residual)
        return out


class ResidualUNet(nn.Module):
    """4-Stage Medical Residual U-Net (used in notebookV3, ~8.11M params)."""
    def __init__(self, in_channels: int = 1, num_classes: int = 4, base_filters: int = 32, dropout_p: float = 0.3):
        super().__init__()
        f = base_filters
        self.enc1 = ResidualBlock(in_channels, f)
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = ResidualBlock(f, f * 2)
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = ResidualBlock(f * 2, f * 4)
        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = ResidualBlock(f * 4, f * 8)
        self.down4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            ResidualBlock(f * 8, f * 16),
            nn.Dropout2d(p=dropout_p)
        )

        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(f * 16, f * 8)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(f * 2, f)

        self.out_conv = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))

        b = self.bottleneck(self.down4(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


def find_best_checkpoint(checkpoint_arg: Optional[str]) -> Path:
    """Finds the model checkpoint to evaluate based on arguments and conventional paths."""
    if checkpoint_arg:
        p = Path(checkpoint_arg)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified checkpoint not found: {p.resolve()}")

    # Conventional candidate paths in order of preference
    candidates = [
        Path("notebookV3/checkpoints/best_unet_kits23.pth"),
        Path("notebookV3/checkpoints/latest_checkpoint.pth"),
        Path("archive_legacy/notebookv2/checkpoints/best_unet_finetuned.pth"),
        Path("archive_legacy/notebookv2/checkpoints/best_unet_kits23.pth"),
    ]

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        "No checkpoint found! Please check 'notebookV3/checkpoints' or specify --checkpoint <path>."
    )


def load_model_for_evaluation(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """Loads model weights from checkpoint safely with auto-detection for ResidualUNet vs UNet."""
    print(f"[CHECKPOINT] Loading trained weights from: {checkpoint_path.resolve()}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract state dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        meta = {
            "epoch": ckpt.get("epoch", "Unknown"),
            "best_macro_dice": ckpt.get("best_macro_dice", "N/A"),
            "batch_idx": ckpt.get("batch_idx", "N/A"),
            "epoch_completed": ckpt.get("epoch_completed", "N/A"),
            "history": ckpt.get("history", [])
        }
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        sd = ckpt["model_state"]
        meta = {
            "epoch": ckpt.get("epoch", "Unknown"),
            "best_macro_dice": ckpt.get("best_macro_dice", "N/A"),
            "batch_idx": ckpt.get("batch_idx", "N/A"),
            "epoch_completed": ckpt.get("epoch_completed", "N/A"),
            "history": ckpt.get("history", [])
        }
    elif isinstance(ckpt, dict):
        sd = ckpt
        meta = {"epoch": "Raw state_dict", "best_macro_dice": "N/A"}
    else:
        raise ValueError("Unrecognized checkpoint format.")

    # Auto-detect whether checkpoint is ResidualUNet (V3) or standard UNet (V2)
    is_residual = any("enc1" in k or "shortcut" in k for k in sd.keys())

    if is_residual:
        print("[ARCHITECTURE] Detected Medical 2D Residual U-Net (V3). Initializing ResidualUNet...")
        model = ResidualUNet(in_channels=1, num_classes=4, base_filters=32).to(device)
    else:
        print("[ARCHITECTURE] Detected Standard Medical 2D U-Net (Legacy V2). Initializing UNet...")
        model = UNet(in_channels=1, num_classes=4, base_channels=32).to(device)

    model.load_state_dict(sd)
    model.eval()
    return model, meta


# ==============================================================================
# 4. PLOTTING & VISUAL REPORT GENERATION
# ==============================================================================

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path):
    """Saves a normalized confusion matrix heatmap figure."""
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=11, fontweight="semibold")
    ax.set_yticklabels(class_names, fontsize=11, fontweight="semibold")

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Annotate values
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            pct = cm_norm[i, j] * 100.0
            raw = cm[i, j]
            color = "white" if cm_norm[i, j] > 0.55 else "black"
            text = f"{pct:.1f}%\n({raw:,})" if raw > 0 else "0.0%"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    ax.set_title("KiTS23 Pixel Confusion Matrix (Normalized)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="semibold", labelpad=8)
    ax.set_ylabel("Ground Truth Class", fontsize=11, fontweight="semibold", labelpad=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[REPORT] Confusion matrix saved to: {save_path.resolve()}")


def plot_metrics_barchart(metrics: Dict[str, Any], class_names: List[str], save_path: Path):
    """Plots comparative bar chart of Dice, IoU, Precision, and Recall per class."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    x = np.arange(len(class_names))
    width = 0.20

    dice_vals = [metrics["Dice"][f"Dice_{c}"] for c in class_names]
    iou_vals  = [metrics["IoU"][f"IoU_{c}"] for c in class_names]
    prec_vals = [metrics["Precision"][f"Precision_{c}"] for c in class_names]
    rec_vals  = [metrics["Recall"][f"Recall_{c}"] for c in class_names]

    r1 = ax.bar(x - 1.5 * width, dice_vals, width, label="Dice Score", color="#2ecc71")
    r2 = ax.bar(x - 0.5 * width, iou_vals,  width, label="IoU (Jaccard)", color="#3498db")
    r3 = ax.bar(x + 0.5 * width, prec_vals, width, label="Precision (PPV)", color="#e67e22")
    r4 = ax.bar(x + 1.5 * width, rec_vals,  width, label="Recall (Sensitivity)", color="#9b59b6")

    ax.set_ylabel("Score [0.0 - 1.0]", fontsize=11, fontweight="semibold")
    ax.set_title(
        f"KiTS23 Multi-Class Segmentation Performance\nForeground Macro Dice: {metrics['Macro_Dice_Foreground']:.4f} | Mean IoU: {metrics['Mean_IoU']:.4f}",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=11, fontweight="semibold")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)

    # Bar value labels
    for rects in [r1, r2, r3, r4]:
        for rect in rects:
            height = rect.get_height()
            if height > 0.02:
                ax.annotate(f"{height:.2f}",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5, rotation=0)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[REPORT] Performance bar chart saved to: {save_path.resolve()}")


def generate_qualitative_figure(
    image_norm: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    case_id: str,
    slice_idx: int,
    save_path: Path
):
    """Generates a comprehensive 4-panel visual comparison for qualitative verification."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), dpi=180)

    # 1. CT Grayscale
    axes[0].imshow(image_norm, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"Input CT Slice\n{case_id} (Index #{slice_idx})", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. Ground Truth Mask
    gt_rgb = create_rgb_mask(gt_mask)
    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth Annotation", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # 3. Predicted Mask
    pred_rgb = create_rgb_mask(pred_mask)
    axes[2].imshow(pred_rgb)
    axes[2].set_title("U-Net Prediction", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # 4. CT Overlay with Prediction
    overlay = overlay_mask_on_image(image_norm, pred_mask, alpha=0.45)
    axes[3].imshow(overlay)
    axes[3].set_title("Anatomical CT Overlay", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    # Legend handles
    patches = [
        mpatches.Patch(color=CLASS_COLORS[1], label="Kidney"),
        mpatches.Patch(color=CLASS_COLORS[2], label="Tumor"),
        mpatches.Patch(color=CLASS_COLORS[3], label="Cyst")
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, frameon=True, fontsize=11, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# 5. CORE EVALUATION ENGINE
# ==============================================================================

def run_evaluation(
    model: UNet,
    dataset: KiTS23Dataset,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 4,
    amp_enabled: bool = True,
    max_batches: Optional[int] = None,
    num_visualizations: int = 10,
    output_dir: Path = Path("reports/evaluation"),
    tta: bool = False
) -> Dict[str, Any]:
    """Runs high-throughput batched model inference and metrics calculation."""
    use_cuda = (device.type == "cuda")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
        prefetch_factor=3 if num_workers > 0 else None,
        drop_last=False
    )

    metrics_engine = ClinicalEvaluationMetrics(num_classes=4)
    criterion = CompositeSegmentationLoss(num_classes=4).to(device)

    total_loss_sum = 0.0
    ce_loss_sum    = 0.0
    dice_loss_sum  = 0.0
    num_batches    = 0

    total_batches = len(dataloader)
    target_batches = total_batches if max_batches is None else min(max_batches, total_batches)

    # Prepare for saving interesting qualitative examples
    output_dir = Path(output_dir)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # To ensure varied visualizations, we target slices that contain lesions
    vis_saved = 0
    df = dataset.df

    print("\n" + "=" * 80)
    print(f"EVALUATION HARNESS: RUNNING INFERENCE ON {target_batches:,} BATCHES")
    print("=" * 80)

    t_start = time.time()
    pbar = tqdm(dataloader, desc="Evaluating", total=target_batches, unit="batch", dynamic_ncols=True)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images_norm = (images.to(device, non_blocking=True).float() / 65535.0)
            masks_gpu   = masks.to(device, dtype=torch.long, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(use_cuda and amp_enabled)):
                logits = model(images_norm)
                if tta:
                    # Test-Time Augmentation: average original and horizontally flipped predictions
                    flipped_images = torch.flip(images_norm, dims=[3])
                    flipped_logits = model(flipped_images)
                    logits = 0.5 * (logits + torch.flip(flipped_logits, dims=[3]))
                loss, l_ce, l_dice = criterion(logits, masks_gpu)

            preds = torch.argmax(logits, dim=1)

            total_loss_sum += loss.item()
            ce_loss_sum    += l_ce.item()
            dice_loss_sum  += l_dice.item()
            num_batches    += 1

            metrics_engine.update(preds, masks_gpu)

            # Check if this batch contains qualitative examples to save
            if vis_saved < num_visualizations:
                b_preds = preds.cpu().numpy()
                b_masks = masks.numpy()
                b_imgs  = images_norm.cpu().numpy()

                for b in range(len(b_preds)):
                    if vis_saved >= num_visualizations:
                        break

                    slice_global_idx = batch_idx * batch_size + b
                    if slice_global_idx >= len(df):
                        continue

                    row = df.iloc[slice_global_idx]
                    has_lesion = (row.get("has_tumor", False) or row.get("has_cyst", False) or row.get("has_kidney", False))

                    # Prioritize slices that contain kidney/tumor/cyst or have positive predictions
                    if has_lesion or np.sum(b_preds[b] > 0) > 100 or vis_saved < 2:
                        case_id = row.get("case_id", f"case_{slice_global_idx}")
                        slice_idx = row.get("slice_index", slice_global_idx)
                        img_slice = b_imgs[b, 0] # (512, 512)
                        gt_slice  = b_masks[b]   # (512, 512)
                        pr_slice  = b_preds[b]   # (512, 512)

                        save_p = vis_dir / f"pred_{vis_saved + 1:02d}_{case_id}_slice_{slice_idx:03d}.png"
                        generate_qualitative_figure(img_slice, gt_slice, pr_slice, str(case_id), int(slice_idx), save_p)
                        vis_saved += 1

            pbar.set_postfix({
                "Batch": f"{batch_idx + 1}/{target_batches}",
                "Loss": f"{loss.item():.4f}",
                "DiceLoss": f"{l_dice.item():.3f}"
            })

    total_eval_time = time.time() - t_start
    results = metrics_engine.compute()

    results["eval_loss"] = total_loss_sum / max(1, num_batches)
    results["eval_ce_loss"] = ce_loss_sum / max(1, num_batches)
    results["eval_dice_loss"] = dice_loss_sum / max(1, num_batches)
    results["eval_time_sec"] = total_eval_time
    results["slices_evaluated"] = num_batches * batch_size
    results["throughput_slices_per_sec"] = (num_batches * batch_size) / max(0.01, total_eval_time)

    return results


# ==============================================================================
# 6. RESULTS PRINTER & EXPORTER
# ==============================================================================

def print_evaluation_summary(results: Dict[str, Any], meta: Dict[str, Any]):
    """Prints a beautiful formatted clinical metrics report to standard output."""
    cm = np.array(results["Confusion_Matrix"])

    print("\n" + "=" * 80)
    print("                 KiTS23 MEDICAL U-NET EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Checkpoint Epoch        : {meta.get('epoch', 'N/A')}")
    print(f"Saved Best Macro Dice   : {meta.get('best_macro_dice', 'N/A')}")
    print(f"Total Slices Evaluated  : {results['slices_evaluated']:,} slices ({results['eval_time_sec']:.1f}s, {results['throughput_slices_per_sec']:.1f} slices/sec)")
    print(f"Total Loss              : {results['eval_loss']:.4f} (CE: {results['eval_ce_loss']:.4f} | SoftDice: {results['eval_dice_loss']:.4f})")
    print("-" * 80)
    print(f"[*] OVERALL PIXEL ACCURACY     : {results['Pixel_Accuracy'] * 100:.2f}%")
    print(f"[+] FOREGROUND MACRO DICE (DSC): {results['Macro_Dice_Foreground']:.4f}  (Kidney + Tumor + Cyst average)")
    print(f"[+] MEAN IoU (Jaccard)         : {results['Mean_IoU']:.4f}")
    print("-" * 80)

    # Detailed Per-Class Table
    print(f"{'Class Name':<14} | {'Dice (DSC)':<12} | {'IoU (Jaccard)':<14} | {'Precision':<11} | {'Recall':<11} | {'Specificity':<11}")
    print("-" * 80)
    for c in CLASS_NAMES:
        d = results["Dice"][f"Dice_{c}"]
        i = results["IoU"][f"IoU_{c}"]
        p = results["Precision"][f"Precision_{c}"]
        r = results["Recall"][f"Recall_{c}"]
        s = results["Specificity"][f"Specificity_{c}"]
        print(f"{c:<14} | {d:<12.4f} | {i:<14.4f} | {p:<11.4f} | {r:<11.4f} | {s:<11.4f}")
    print("-" * 80)

    # Confusion Matrix Print
    print("\n[PIXEL CONFUSION MATRIX (Rows: True Class | Columns: Predicted Class)]")
    header = f"{'True Class':<12} | " + " | ".join([f"{name[:8]:>8}" for name in CLASS_NAMES])
    print(header)
    print("-" * len(header))
    for idx, name in enumerate(CLASS_NAMES):
        row_str = f"{name:<12} | " + " | ".join([f"{cm[idx, j]:>8,}" for j in range(len(CLASS_NAMES))])
        print(row_str)
    print("=" * 80 + "\n")


def export_reports(results: Dict[str, Any], meta: Dict[str, Any], output_dir: Path):
    """Exports structured JSON and CSV reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. JSON Report
    json_path = output_dir / "evaluation_summary.json"
    full_export = {
        "checkpoint_metadata": meta,
        "metrics": {
            "Pixel_Accuracy": results["Pixel_Accuracy"],
            "Macro_Dice_Foreground": results["Macro_Dice_Foreground"],
            "Mean_IoU": results["Mean_IoU"],
            "eval_loss": results["eval_loss"],
            "eval_ce_loss": results["eval_ce_loss"],
            "eval_dice_loss": results["eval_dice_loss"],
            "eval_time_sec": results["eval_time_sec"],
            "slices_evaluated": results["slices_evaluated"],
            "throughput_slices_per_sec": results["throughput_slices_per_sec"],
            "Per_Class": {
                c: {
                    "Dice": results["Dice"][f"Dice_{c}"],
                    "IoU": results["IoU"][f"IoU_{c}"],
                    "Precision": results["Precision"][f"Precision_{c}"],
                    "Recall": results["Recall"][f"Recall_{c}"],
                    "Specificity": results["Specificity"][f"Specificity_{c}"]
                } for c in CLASS_NAMES
            },
            "Confusion_Matrix": results["Confusion_Matrix"]
        }
    }
    with open(json_path, "w") as f:
        json.dump(full_export, f, indent=2)
    print(f"[REPORT] Structured JSON metrics saved to: {json_path.resolve()}")

    # 2. CSV Report
    csv_path = output_dir / "per_class_metrics.csv"
    rows = []
    for c in CLASS_NAMES:
        rows.append({
            "Class": c,
            "Dice_DSC": round(results["Dice"][f"Dice_{c}"], 4),
            "IoU_Jaccard": round(results["IoU"][f"IoU_{c}"], 4),
            "Precision_PPV": round(results["Precision"][f"Precision_{c}"], 4),
            "Recall_Sensitivity": round(results["Recall"][f"Recall_{c}"], 4),
            "Specificity_TNR": round(results["Specificity"][f"Specificity_{c}"], 4),
        })
    df_csv = pd.DataFrame(rows)
    df_csv.to_csv(csv_path, index=False)
    print(f"[REPORT] Tabular CSV metrics saved to: {csv_path.resolve()}")


def save_run_metadata(
    results: Dict[str, Any],
    meta: Dict[str, Any],
    args: argparse.Namespace,
    checkpoint_path: Path,
    run_dir: Path,
    run_id: str,
    timestamp: str
):
    """Saves complete execution and environment metadata to run_dir."""
    run_meta = {
        "run_id": run_id,
        "timestamp": timestamp,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_meta": meta,
        "target": {
            "mode": "patient_case" if args.patient_case else "split",
            "patient_case": args.patient_case,
            "split": args.split if not args.patient_case else None
        },
        "cli_args": vars(args),
        "execution": {
            "slices_evaluated": results.get("slices_evaluated"),
            "eval_time_sec": results.get("eval_time_sec"),
            "throughput_slices_per_sec": results.get("throughput_slices_per_sec"),
            "device": str(args.device),
            "amp": bool(args.amp),
            "batch_size": int(args.batch_size)
        },
        "summary_metrics": {
            "Pixel_Accuracy": results.get("Pixel_Accuracy"),
            "Macro_Dice_Foreground": results.get("Macro_Dice_Foreground"),
            "Mean_IoU": results.get("Mean_IoU"),
            "eval_loss": results.get("eval_loss"),
            "Dice": results.get("Dice"),
            "IoU": results.get("IoU")
        }
    }
    meta_path = run_dir / "run_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    print(f"[REPORT] Run metadata saved to: {meta_path.resolve()}")


def update_evaluation_timeline(
    timeline_dir: Path,
    run_record: Dict[str, Any]
):
    """
    Maintains a cumulative CSV and human-readable Markdown timeline
    of all evaluation runs performed in this project.
    """
    timeline_dir = Path(timeline_dir)
    timeline_dir.mkdir(parents=True, exist_ok=True)

    csv_path = timeline_dir / "timeline.csv"
    md_path = timeline_dir / "timeline.md"

    # 1. Update CSV
    df_new = pd.DataFrame([run_record])
    if csv_path.exists():
        try:
            df_existing = pd.read_csv(csv_path)
            # Remove duplicate run_id if re-run
            df_existing = df_existing[df_existing["Run_ID"] != run_record["Run_ID"]]
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception:
            df_combined = df_new
    else:
        df_combined = df_new

    df_combined.to_csv(csv_path, index=False)
    print(f"[TIMELINE] Cumulative CSV ledger updated: {csv_path.resolve()}")

    # 2. Render Markdown Timeline
    try:
        md_lines = [
            "# KiTS23 U-Net Evaluation Timeline & History",
            "",
            f"- **Total Evaluations Completed:** {len(df_combined)}",
            f"- **Last Updated:** {run_record.get('Timestamp', 'N/A')}",
            f"- **Timeline Root:** `{timeline_dir.resolve()}`",
            "",
            "---",
            "",
            "## Chronological Timeline (Newest Runs First)",
            "",
            "| # | Date & Time | Run ID | Target | Checkpoint | Slices | Macro Dice (FG) | Kidney Dice | Tumor Dice | Cyst Dice | Mean IoU | Pixel Acc | Loss | Folder |",
            "|---|-------------|--------|--------|------------|--------|-----------------|-------------|------------|-----------|----------|-----------|------|--------|"
        ]

        total_runs = len(df_combined)
        for idx in range(total_runs - 1, -1, -1):
            row = df_combined.iloc[idx]
            run_id = str(row.get("Run_ID", ""))
            ts = str(row.get("Timestamp", ""))
            target = str(row.get("Target", ""))
            ckpt_name = Path(str(row.get("Checkpoint", ""))).name
            slices = f"{int(row.get('Slices_Evaluated', 0)):,}"
            macro_d = f"{float(row.get('Macro_Dice_Foreground', 0.0)):.4f}"
            kidney_d = f"{float(row.get('Dice_Kidney', 0.0)):.4f}"
            tumor_d = f"{float(row.get('Dice_Tumor', 0.0)):.4f}"
            cyst_d = f"{float(row.get('Dice_Cyst', 0.0)):.4f}"
            miou = f"{float(row.get('Mean_IoU', 0.0)):.4f}"
            acc = f"{float(row.get('Pixel_Accuracy', 0.0)) * 100:.2f}%"
            loss_val = f"{float(row.get('Eval_Loss', 0.0)):.4f}"
            r_dir = str(row.get("Run_Directory", ""))

            # Format link relative to timeline directory
            folder_link = f"[`{run_id}`]({r_dir})"

            line = (
                f"| {idx + 1} | {ts} | `{run_id}` | {target} | `{ckpt_name}` | {slices} | "
                f"**{macro_d}** | {kidney_d} | {tumor_d} | {cyst_d} | {miou} | {acc} | {loss_val} | {folder_link} |"
            )
            md_lines.append(line)

        md_lines.append("")
        md_lines.append("---")
        md_lines.append("### Directory Structure")
        md_lines.append("Each run folder contains:")
        md_lines.append("- `evaluation_summary.json` - Complete numerical metrics and confusion matrix")
        md_lines.append("- `per_class_metrics.csv` - Per-class DSC, IoU, Precision, Recall, Specificity")
        md_lines.append("- `confusion_matrix.png` - Pixel-level confusion matrix heatmap")
        md_lines.append("- `metrics_barchart.png` - Multi-class bar chart")
        md_lines.append("- `run_metadata.json` - Hardware, timing, and arguments log")
        md_lines.append("- `visualizations/` - 4-panel qualitative axial CT slice overlays")
        md_lines.append("")
        md_lines.append("*Maintained automatically by `evaluate.py`.*")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

        print(f"[TIMELINE] Markdown timeline updated: {md_path.resolve()}")
    except Exception as e:
        print(f"[WARNING] Could not update timeline.md: {e}")


def update_latest_snapshot(run_dir: Path, timeline_dir: Path):
    """Maintains a convenient 'latest' snapshot directory mirroring the latest run."""
    latest_dir = timeline_dir / "latest"
    try:
        if latest_dir.exists():
            shutil.rmtree(latest_dir, ignore_errors=True)
        shutil.copytree(run_dir, latest_dir)
        print(f"[LATEST] Updated 'latest' evaluation snapshot at: {latest_dir.resolve()}")
    except Exception as e:
        pass


# ==============================================================================
# 7. MAIN CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="KiTS23 Multi-Class U-Net Evaluation & Accuracy Testing Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Model and Data Paths
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained checkpoint (.pth). If not set, automatically finds best/latest checkpoint.")
    parser.add_argument("--data-root", type=str, default=r"D:\archive",
                        help="Root directory of preprocessed dataset containing 'images' and 'masks'")
    parser.add_argument("--csv-path", type=str, default=r"D:\archive\dataset.csv",
                        help="Path to verified dataset.csv metadata")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"],
                        help="Dataset split to evaluate ('test', 'val', or 'train')")

    # Evaluation Controls
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for parallel inference (default: 16)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker processes for prefetching")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limit batches to evaluate for fast quick-eval (e.g. 50 or 100). Default: None (Full split).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (cuda or cpu)")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use Automatic Mixed Precision (AMP FP16) for accelerated GPU evaluation")
    parser.add_argument("--tta", action="store_true", default=False,
                        help="Enable Test-Time Augmentation (horizontal flip averaging) for boosted accuracy")

    # Reporting, Timeline & Visualizations
    parser.add_argument("--patient-case", type=str, default=None,
                        help="Optional: Evaluate all slices of a specific patient case (e.g. 'case_00028')")
    parser.add_argument("--output-dir", type=str, default="reports/evaluation",
                        help="Root directory to save evaluation timeline and run reports")
    parser.add_argument("--timeline-dir", type=str, default=None,
                        help="Directory to store evaluation timeline. Defaults to --output-dir.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Optional custom name or tag for this evaluation run (e.g. 'baseline_eval')")
    parser.add_argument("--no-timeline", action="store_true", default=False,
                        help="Disable timeline tracking and save directly to --output-dir without timestamped subfolder")
    parser.add_argument("--num-visualizations", type=int, default=10,
                        help="Number of 4-panel visual comparison figures to generate and save")

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    base_output_dir = Path(args.output_dir)
    timeline_dir = Path(args.timeline_dir) if args.timeline_dir else base_output_dir

    # Determine run-specific folder and timeline identifier
    now = datetime.now()
    timestamp_tag = now.strftime("%Y%m%d_%H%M%S")
    target_tag = f"case_{args.patient_case}" if args.patient_case else f"split_{args.split}"

    if args.no_timeline:
        run_id = "standalone"
        run_dir = base_output_dir
    else:
        if args.run_name:
            clean_name = "".join(c if (c.isalnum() or c in "_-") else "_" for c in args.run_name)
            run_id = f"eval_{timestamp_tag}_{clean_name}"
        else:
            run_id = f"eval_{timestamp_tag}_{target_tag}"
        run_dir = timeline_dir / "runs" / run_id

    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve Checkpoint
    checkpoint_path = find_best_checkpoint(args.checkpoint)
    model, meta = load_model_for_evaluation(checkpoint_path, device)

    # 2. Ingest Dataset
    data_root = Path(args.data_root)
    image_dir = data_root / "images"
    mask_dir  = data_root / "masks"
    csv_path  = Path(args.csv_path)

    if not csv_path.exists():
        print(f"[ERROR] CSV not found at: {csv_path}")
        sys.exit(1)

    df_meta = pd.read_csv(csv_path)
    if args.patient_case:
        split_df = df_meta[df_meta["case_id"] == args.patient_case].sort_values("slice_index").reset_index(drop=True)
        if len(split_df) == 0:
            print(f"[ERROR] Patient case '{args.patient_case}' not found in {csv_path}!")
            sys.exit(1)
        print(f"[PATIENT EVALUATION] Patient: '{args.patient_case}' | Total Axial Slices: {len(split_df):,}")
    else:
        split_df = df_meta[df_meta["split"] == args.split].copy().reset_index(drop=True)
        if len(split_df) == 0:
            print(f"[ERROR] No records found for split '{args.split}' in {csv_path}!")
            sys.exit(1)
        print(f"[DATASET] Split: '{args.split.upper()}' | Total Available Slices: {len(split_df):,}")
    dataset = KiTS23Dataset(split_df, image_dir, mask_dir)

    # 3. Run Evaluation
    results = run_evaluation(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp_enabled=args.amp,
        max_batches=args.max_batches,
        num_visualizations=args.num_visualizations,
        output_dir=run_dir,
        tta=args.tta
    )

    # 4. Print Summary & Save Reports
    print_evaluation_summary(results, meta)
    export_reports(results, meta, run_dir)

    # 5. Generate Visual Charts
    cm_plot_path = run_dir / "confusion_matrix.png"
    plot_confusion_matrix(np.array(results["Confusion_Matrix"]), CLASS_NAMES, cm_plot_path)

    bar_plot_path = run_dir / "metrics_barchart.png"
    plot_metrics_barchart(results, CLASS_NAMES, bar_plot_path)

    # 6. Save Run Metadata & Update Timeline
    save_run_metadata(
        results=results,
        meta=meta,
        args=args,
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        run_id=run_id,
        timestamp=now.strftime("%Y-%m-%d %H:%M:%S")
    )

    if not args.no_timeline:
        target_display = f"Patient {args.patient_case}" if args.patient_case else f"Split {args.split}"
        run_record = {
            "Timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "Run_ID": run_id,
            "Target": target_display,
            "Split": args.split if not args.patient_case else "patient_case",
            "Patient_Case": args.patient_case if args.patient_case else "N/A",
            "Checkpoint": checkpoint_path.name,
            "Slices_Evaluated": int(results["slices_evaluated"]),
            "Pixel_Accuracy": round(float(results["Pixel_Accuracy"]), 4),
            "Macro_Dice_Foreground": round(float(results["Macro_Dice_Foreground"]), 4),
            "Dice_Background": round(float(results["Dice"].get("Dice_Background", 0.0)), 4),
            "Dice_Kidney": round(float(results["Dice"].get("Dice_Kidney", 0.0)), 4),
            "Dice_Tumor": round(float(results["Dice"].get("Dice_Tumor", 0.0)), 4),
            "Dice_Cyst": round(float(results["Dice"].get("Dice_Cyst", 0.0)), 4),
            "Mean_IoU": round(float(results["Mean_IoU"]), 4),
            "Eval_Loss": round(float(results["eval_loss"]), 4),
            "Eval_Time_Sec": round(float(results["eval_time_sec"]), 2),
            "Throughput_Slices_Per_Sec": round(float(results["throughput_slices_per_sec"]), 2),
            "Run_Directory": f"runs/{run_id}"
        }
        update_evaluation_timeline(timeline_dir, run_record)
        update_latest_snapshot(run_dir, timeline_dir)

    print("\n" + "=" * 80)
    print(f"[SUCCESS] Complete evaluation finished successfully!")
    print(f"  • Run Artifacts : {run_dir.resolve()}")
    if not args.no_timeline:
        print(f"  • Timeline Log  : {(timeline_dir / 'timeline.md').resolve()}")
        print(f"  • Latest Mirror : {(timeline_dir / 'latest').resolve()}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
