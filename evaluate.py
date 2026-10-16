#!/usr/bin/env python3
"""
================================================================================
KiTS23 2D U-Net / Residual U-Net CLI Evaluation & Clinical Metrics Engine
================================================================================

A simple, standalone command-line evaluation tool for KiTS23 kidney tumor
segmentation models.

Features:
  - Simple, zero-config CLI: Just run `python evaluate.py` to evaluate the best
    model on the test split.
  - Timeline-based folder organization: Automatically saves each run's reports,
    metrics, charts, and overlay visualizations into a timestamped directory
    (e.g., `reports/timeline/2026-10-14_08-30-00_test/`).
  - Cumulative Timeline Ledger: Automatically tracks and updates a chronological
    Markdown log (`timeline.md`) and CSV database (`timeline.csv`).
  - Standalone & self-contained: No external training dependencies; includes
    architecture definitions (ResidualUNet and legacy UNet) and data loaders.
  - Comprehensive clinical metrics:
      * Overall Pixel Accuracy
      * Per-Class Dice Similarity Coefficient (DSC): Background, Kidney, Tumor, Cyst
      * Foreground Macro Dice (KiTS Official Benchmark: Kidney + Tumor + Cyst)
      * Per-Class IoU (Jaccard Index), Precision (PPV), Recall (Sensitivity), Specificity
      * 4x4 Pixel Confusion Matrix
  - High-resolution visual artifacts:
      * Confusion matrix heatmap
      * Multi-metric performance bar chart
      * 4-panel axial CT qualitative overlays (CT, Ground Truth, Prediction, Blend)
      * Markdown clinical scorecard & JSON / CSV exports
================================================================================
"""

import sys
import os
import time
import json
import shutil
import argparse
import platform
import pathlib
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

# Cross-platform compatibility fix for checkpoints serialized on Windows
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# Safeguard Windows console text encoding
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
from scipy.ndimage import label

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / CLI execution
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ==============================================================================
# 1. CONSTANTS, COLORMAPS & SPATIAL STANDARDIZATION
# ==============================================================================

CLASS_NAMES = ["Background", "Kidney", "Tumor", "Cyst"]

# Standard KiTS Anatomical RGB Palette (normalized 0.0 - 1.0)
CLASS_COLORS = {
    0: (0.0, 0.0, 0.0),        # Background: Black
    1: (0.06, 0.73, 0.51),      # Kidney: Emerald Green (#10B981)
    2: (0.94, 0.27, 0.27),      # Tumor: Crimson Red (#EF4444)
    3: (0.23, 0.51, 0.96),      # Cyst: Cerulean Blue (#3B82F6)
}


def standardize_spatial_dims(
    image: np.ndarray,
    mask: np.ndarray,
    target_h: int = 512,
    target_w: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Center-crops or zero-pads matched 2D image and mask arrays to fixed (target_h, target_w)
    dimensions without distorting anatomical aspect ratios.
    """
    h, w = image.shape[:2]
    if h == target_h and w == target_w:
        return image, mask

    # Height adjustment
    if h > target_h:
        top = (h - target_h) // 2
        image, mask = image[top:top + target_h, :], mask[top:top + target_h, :]
    elif h < target_h:
        top = (target_h - h) // 2
        bottom = target_h - h - top
        image = np.pad(image, ((top, bottom), (0, 0)), mode="constant", constant_values=0)
        mask = np.pad(mask, ((top, bottom), (0, 0)), mode="constant", constant_values=0)

    # Width adjustment
    if w > target_w:
        left = (w - target_w) // 2
        image, mask = image[:, left:left + target_w], mask[:, left:left + target_w]
    elif w < target_w:
        left = (target_w - w) // 2
        right = target_w - w - left
        image = np.pad(image, ((0, 0), (left, right)), mode="constant", constant_values=0)
        mask = np.pad(mask, ((0, 0), (left, right)), mode="constant", constant_values=0)

    return image, mask


def create_rgb_mask(mask: np.ndarray) -> np.ndarray:
    """Converts a 2D categorical class mask into an (H, W, 3) RGB color mask."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for c, color in CLASS_COLORS.items():
        if c == 0:
            continue
        rgb[mask == c] = color
    return rgb


def overlay_mask_on_image(image_norm: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blends color-coded prediction mask onto grayscale normalized CT image [0, 1]."""
    h, w = image_norm.shape
    img_rgb = np.stack([image_norm] * 3, axis=-1)
    rgb_mask = create_rgb_mask(mask)
    fg = (mask > 0)

    blended = img_rgb.copy()
    blended[fg] = (1.0 - alpha) * img_rgb[fg] + alpha * rgb_mask[fg]
    return np.clip(blended, 0.0, 1.0)


# ==============================================================================
# 2. DATASET DEFINITION
# ==============================================================================

class KiTS23Dataset(Dataset):
    """
    High-performance dataset reader for 16-bit CT slice images and 8-bit masks.
    """
    def __init__(self, df: pd.DataFrame, image_dir: Path, mask_dir: Path):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.filenames = self.df["filename"].to_numpy()
        self.case_ids = self.df["case_id"].to_numpy()
        self.img_paths = [str(self.image_dir / f) for f in self.filenames]
        self.msk_paths = [str(self.mask_dir / f) for f in self.filenames]

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.img_paths[idx]) as p_img:
            img_np = np.array(p_img, dtype=np.int32)
        with Image.open(self.msk_paths[idx]) as p_msk:
            msk_np = np.array(p_msk, dtype=np.uint8)

        img_np, msk_np = standardize_spatial_dims(img_np, msk_np, 512, 512)
        img_t = torch.from_numpy(img_np).unsqueeze(0)
        msk_t = torch.from_numpy(msk_np).long()
        return img_t, msk_t


# ==============================================================================
# 3. SELF-CONTAINED MODEL ARCHITECTURES
# ==============================================================================

class ResidualBlock(nn.Module):
    """Two-convolution residual block with InstanceNorm2d and PReLU activation."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.act1 = nn.PReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.act2 = nn.PReLU()

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
    """Medical 2D Residual U-Net (used in notebookV3 and production backend, ~8.11M params)."""
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


# Legacy U-Net components for backward-compatibility with older v2 checkpoints
class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Standard Medical 2D U-Net (Legacy V2 model)."""
    def __init__(self, in_channels: int = 1, num_classes: int = 4, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)
        self.up1 = Up(c * 16 + c * 8, c * 8)
        self.up2 = Up(c * 8 + c * 4, c * 4)
        self.up3 = Up(c * 4 + c * 2, c * 2)
        self.up4 = Up(c * 2 + c, c)
        self.outc = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class CompositeLoss(nn.Module):
    """Cross-Entropy + Multi-Class Soft Dice Loss."""
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ce_loss = F.cross_entropy(logits, targets)
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).contiguous().float()

        probs_flat = probs.view(probs.size(0), self.num_classes, -1)
        targets_flat = targets_one_hot.view(targets_one_hot.size(0), self.num_classes, -1)

        intersection = (probs_flat * targets_flat).sum(dim=2)
        cardinality = (probs_flat**2 + targets_flat**2).sum(dim=2)

        dice_per_class = (2.0 * intersection + 1e-5) / (cardinality + 1e-5)
        dice_loss = 1.0 - dice_per_class.mean()

        total_loss = ce_loss + dice_loss
        return total_loss, ce_loss, dice_loss


# ==============================================================================
# 4. CLINICAL EVALUATION METRICS ENGINE
# ==============================================================================

class ClinicalEvaluationMetrics:
    """
    Computes rigorous pixel-level and per-class clinical statistics:
      - 4x4 Confusion Matrix (rows = Ground Truth, cols = Prediction)
      - Per-class Dice, IoU, Precision, Recall, Specificity
      - Foreground Macro Dice, Mean IoU, Overall Pixel Accuracy
    """
    def __init__(self, num_classes: int = 4, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or CLASS_NAMES
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_pixels = 0
        self.total_correct = 0

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        p = preds.detach().cpu().numpy().reshape(-1)
        t = targets.detach().cpu().numpy().reshape(-1)

        self.total_pixels += p.size
        self.total_correct += int(np.sum(p == t))

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

            # Dice (DSC / F1)
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

        # Macro Foreground Dice (Kidney + Tumor + Cyst)
        fg_dice_values = [dice_per_class[f"Dice_{name}"] for name in self.class_names[1:]]
        macro_dice = float(np.mean(fg_dice_values))

        # Mean IoU across all 4 classes
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
# 5. CHECKPOINT LOADER & AUTO-DISCOVERY
# ==============================================================================

def find_best_checkpoint(checkpoint_arg: Optional[str]) -> Path:
    """Finds model checkpoint based on explicit argument or prioritized search candidates."""
    if checkpoint_arg:
        p = Path(checkpoint_arg)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified checkpoint not found: {p.resolve()}")

    candidates = [
        Path("notebookV3/checkpoints/best_model_retrain_tversky_loss.pth"),
        Path("notebookV3/checkpoints/best_unet_kits23.pth"),
        Path("notebookV3/checkpointsV2/resunet_fullval_v2/best_model.pth"),
        Path("notebookV3/checkpointsV2/resunet_fullval_v2/latest_model.pth"),
        Path("notebookV3/checkpoints/latest_checkpoint.pth"),
        Path("archive_legacy/notebookv2/checkpoints/best_unet_finetuned.pth"),
        Path("archive_legacy/notebookv2/checkpoints/best_unet_kits23.pth"),
    ]

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        "No model checkpoint (.pth) found in conventional locations!\n"
        "Please provide the path using `--checkpoint <path>`."
    )


def auto_detect_dataset(data_root_arg: Optional[str], csv_path_arg: Optional[str]) -> Tuple[Path, Path]:
    """Resolves data root directory and dataset.csv path automatically."""
    candidates = [
        Path(data_root_arg) if data_root_arg else None,
        Path(r"D:\archive"),
        Path("archive"),
        Path("../archive"),
    ]

    root = None
    for cand in candidates:
        if cand and cand.exists() and (cand / "images").exists() and (cand / "masks").exists():
            root = cand.resolve()
            break

    if root is None:
        raise FileNotFoundError(
            "Dataset folder with 'images' and 'masks' subdirectories not found!\n"
            "Please specify using `--data-root <path>`."
        )

    if csv_path_arg:
        csv_path = Path(csv_path_arg).resolve()
    else:
        csv_path = root / "dataset.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file 'dataset.csv' not found at: {csv_path}")

    return root, csv_path


def load_model_for_evaluation(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """Loads weights with automatic architecture detection (ResidualUNet vs UNet)."""
    print(f"[CHECKPOINT] Loading weights from: {checkpoint_path.resolve()}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    meta = {}
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        meta = {
            "epoch": ckpt.get("epoch", "Unknown"),
            "best_macro_dice": ckpt.get("best_macro_dice", "N/A"),
            "batch_idx": ckpt.get("batch_idx", "N/A"),
        }
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        sd = ckpt["model_state"]
        meta = {
            "epoch": ckpt.get("epoch", "Unknown"),
            "best_macro_dice": ckpt.get("best_macro_dice", "N/A"),
            "batch_idx": ckpt.get("batch_idx", "N/A"),
        }
    elif isinstance(ckpt, dict):
        sd = ckpt
        meta = {"epoch": "Direct State Dict", "best_macro_dice": "N/A"}
    else:
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")

    # Inspect keys to determine architecture
    is_residual = any("enc1" in k or "shortcut" in k for k in sd.keys())

    if is_residual:
        print("[ARCHITECTURE] Detected Medical 2D Residual U-Net (V3 / Production).")
        model = ResidualUNet(in_channels=1, num_classes=4, base_filters=32).to(device)
    else:
        print("[ARCHITECTURE] Detected Standard Medical 2D U-Net (Legacy V2).")
        model = UNet(in_channels=1, num_classes=4, base_channels=32).to(device)

    model.load_state_dict(sd)
    model.eval()
    return model, meta


# ==============================================================================
# 6. PLOTTING & VISUALIZATION GENERATORS
# ==============================================================================

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path):
    """Generates and saves a clean normalized confusion matrix heatmap."""
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=180)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=11, fontweight="semibold")
    ax.set_yticklabels(class_names, fontsize=11, fontweight="semibold")

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

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


def plot_metrics_barchart(metrics: Dict[str, Any], class_names: List[str], save_path: Path):
    """Generates comparative multi-class bar chart across clinical metrics."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    x = np.arange(len(class_names))
    width = 0.20

    dice_vals = [metrics["Dice"][f"Dice_{c}"] for c in class_names]
    iou_vals = [metrics["IoU"][f"IoU_{c}"] for c in class_names]
    prec_vals = [metrics["Precision"][f"Precision_{c}"] for c in class_names]
    rec_vals = [metrics["Recall"][f"Recall_{c}"] for c in class_names]

    r1 = ax.bar(x - 1.5 * width, dice_vals, width, label="Dice (DSC)", color="#10B981")
    r2 = ax.bar(x - 0.5 * width, iou_vals, width, label="IoU (Jaccard)", color="#3B82F6")
    r3 = ax.bar(x + 0.5 * width, prec_vals, width, label="Precision (PPV)", color="#F59E0B")
    r4 = ax.bar(x + 1.5 * width, rec_vals, width, label="Recall (Sens.)", color="#8B5CF6")

    ax.set_ylabel("Score [0.0 - 1.0]", fontsize=11, fontweight="semibold")
    ax.set_title(
        f"KiTS23 Segmentation Performance by Anatomical Class\n"
        f"Foreground Macro Dice: {metrics['Macro_Dice_Foreground']:.4f} | Mean IoU: {metrics['Mean_IoU']:.4f} | Pixel Acc: {metrics['Pixel_Accuracy']*100:.2f}%",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=11, fontweight="semibold")
    ax.set_ylim(0.0, 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)

    for rects in [r1, r2, r3, r4]:
        for rect in rects:
            height = rect.get_height()
            if height > 0.02:
                ax.annotate(f"{height:.2f}",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def generate_qualitative_figure(
    image_norm: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    case_id: str,
    slice_idx: int,
    save_path: Path
):
    """Generates 4-panel axial slice comparison (CT, Ground Truth, Prediction, Blend)."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), dpi=160)

    # 1. Grayscale CT
    axes[0].imshow(image_norm, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"Input Axial CT\n{case_id} (Slice #{slice_idx})", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. Ground Truth
    gt_rgb = create_rgb_mask(gt_mask)
    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth Annotation", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # 3. Model Prediction
    pred_rgb = create_rgb_mask(pred_mask)
    axes[2].imshow(pred_rgb)
    axes[2].set_title("Model Prediction", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # 4. CT Overlay with Prediction
    overlay = overlay_mask_on_image(image_norm, pred_mask, alpha=0.45)
    axes[3].imshow(overlay)
    axes[3].set_title("Prediction CT Overlay", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    # Legend
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
# 7. CORE INFERENCE & POST-PROCESSING ENGINE
# ==============================================================================

def apply_prediction_postprocessing(
    logits: torch.Tensor,
    calibrated: bool = False,
    tumor_thresh: float = 0.35,
    cyst_thresh: float = 0.28,
    kidney_thresh: float = 0.50,
    anatomical_containment: bool = False,
    min_kidney_voxels: int = 0,
    min_tumor_voxels: int = 0,
    min_cyst_voxels: int = 0,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Applies calibrated probability thresholding, GPU anatomical envelope containment,
    and morphological connected component noise filtering.
    """
    B, C, H, W = logits.shape
    probs = F.softmax(logits, dim=1)

    if calibrated:
        # Cost-sensitive decision rule:
        # Overcomes background prior probability dominance and recovers under-segmented lesions
        preds = torch.zeros((B, H, W), dtype=torch.long, device=device)
        p_k = probs[:, 1]
        p_t = probs[:, 2]
        p_c = probs[:, 3]

        preds[p_k >= kidney_thresh] = 1
        preds[(p_t >= tumor_thresh) & (p_t >= p_c)] = 2
        preds[(p_c >= cyst_thresh) & (p_c >= p_t)] = 3
    else:
        preds = torch.argmax(logits, dim=1)

    if anatomical_containment:
        # GPU-accelerated morphological envelope containment using F.max_pool2d
        # Renal tumors and cysts must be anatomically anchored to the retroperitoneum/kidney
        k_t_mask = ((preds == 1) | (preds == 2)).unsqueeze(1).float()
        has_organ = (k_t_mask.view(B, -1).sum(dim=1) > 20)  # at least 20 voxels on slice

        wide_env = F.max_pool2d(k_t_mask, kernel_size=51, stride=1, padding=25).squeeze(1) > 0.5
        tight_env = F.max_pool2d(k_t_mask, kernel_size=25, stride=1, padding=12).squeeze(1) > 0.5

        organ_mask = has_organ.view(B, 1, 1)
        # Suppress tumor outside wide envelope if kidney exists on slice
        preds[(preds == 2) & organ_mask & (~wide_env)] = 0
        # Revert stray cyst outside tight envelope to kidney
        preds[(preds == 3) & organ_mask & (~tight_env)] = 1
        # Suppress any tumor or cyst on non-renal slices
        preds[(~organ_mask) & (preds > 0)] = 0

    if min_kidney_voxels > 0 or min_tumor_voxels > 0 or min_cyst_voxels > 0:
        preds_np = preds.cpu().numpy()
        for i in range(B):
            slice_p = preds_np[i]
            if min_kidney_voxels > 0 and np.any(slice_p == 1):
                lbl, num = label(slice_p == 1)
                for l_idx in range(1, num + 1):
                    if np.sum(lbl == l_idx) < min_kidney_voxels:
                        slice_p[lbl == l_idx] = 0
            if min_tumor_voxels > 0 and np.any(slice_p == 2):
                lbl, num = label(slice_p == 2)
                for l_idx in range(1, num + 1):
                    if np.sum(lbl == l_idx) < min_tumor_voxels:
                        slice_p[lbl == l_idx] = 0
            if min_cyst_voxels > 0 and np.any(slice_p == 3):
                lbl, num = label(slice_p == 3)
                for l_idx in range(1, num + 1):
                    if np.sum(lbl == l_idx) < min_cyst_voxels:
                        slice_p[lbl == l_idx] = 1  # reassign tiny isolated cyst speckle to surrounding kidney
            preds_np[i] = slice_p
        preds = torch.from_numpy(preds_np).to(device)

    return preds


def run_evaluation(
    model: nn.Module,
    dataset: KiTS23Dataset,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 4,
    amp_enabled: bool = True,
    max_batches: Optional[int] = None,
    num_visualizations: int = 10,
    output_dir: Path = Path("reports/timeline"),
    tta: bool = False,
    calibrated: bool = False,
    tumor_thresh: float = 0.35,
    cyst_thresh: float = 0.28,
    kidney_thresh: float = 0.50,
    anatomical_containment: bool = False,
    min_kidney_voxels: int = 0,
    min_tumor_voxels: int = 0,
    min_cyst_voxels: int = 0
) -> Dict[str, Any]:
    """Runs high-throughput batched inference, metric computation, and visualization export."""
    # Graceful DataLoader worker handling for Windows
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0)
        )
        # Test iterator creation
        _ = iter(dataloader)
    except Exception:
        print("[DATA INFO] Adjusting num_workers to 0 for Windows compatibility...")
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda")
        )

    criterion = CompositeLoss(num_classes=4)
    metrics_engine = ClinicalEvaluationMetrics(num_classes=4, class_names=CLASS_NAMES)
    use_cuda = (device.type == "cuda")

    total_loss_sum = 0.0
    ce_loss_sum = 0.0
    dice_loss_sum = 0.0
    num_batches = 0

    total_batches = len(dataloader)
    target_batches = total_batches if max_batches is None else min(max_batches, total_batches)

    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    vis_saved = 0
    df = dataset.df

    postproc_desc = []
    if calibrated:
        postproc_desc.append(f"Calibrated Thresholds (T:{tumor_thresh:.2f}, C:{cyst_thresh:.2f}, K:{kidney_thresh:.2f})")
    if anatomical_containment:
        postproc_desc.append("Anatomical Envelope Containment")
    if min_tumor_voxels > 0 or min_cyst_voxels > 0 or min_kidney_voxels > 0:
        postproc_desc.append(f"CC Filter (K:{min_kidney_voxels}, T:{min_tumor_voxels}, C:{min_cyst_voxels})")
    if not postproc_desc:
        postproc_desc.append("Standard Argmax")

    print("\n" + "=" * 80)
    print(f"RUNNING INFERENCE ON {target_batches:,} BATCHES ({target_batches * batch_size:,} SLICES MAX)")
    print(f"Hardware: {device.type.upper()} | AMP FP16: {amp_enabled and use_cuda} | TTA: {tta}")
    print(f"Post-Processing Engine : {' | '.join(postproc_desc)}")
    print("=" * 80)

    t_start = time.time()
    pbar = tqdm(dataloader, desc="Evaluating", total=target_batches, unit="batch", dynamic_ncols=True)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images_norm = (images.to(device, non_blocking=True).float() / 65535.0)
            masks_gpu = masks.to(device, dtype=torch.long, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(use_cuda and amp_enabled)):
                logits = model(images_norm)
                if tta:
                    flipped_images = torch.flip(images_norm, dims=[3])
                    flipped_logits = model(flipped_images)
                    logits = 0.5 * (logits + torch.flip(flipped_logits, dims=[3]))
                loss, l_ce, l_dice = criterion(logits, masks_gpu)

            preds = apply_prediction_postprocessing(
                logits=logits,
                calibrated=calibrated,
                tumor_thresh=tumor_thresh,
                cyst_thresh=cyst_thresh,
                kidney_thresh=kidney_thresh,
                anatomical_containment=anatomical_containment,
                min_kidney_voxels=min_kidney_voxels,
                min_tumor_voxels=min_tumor_voxels,
                min_cyst_voxels=min_cyst_voxels,
                device=device
            )

            total_loss_sum += loss.item()
            ce_loss_sum += l_ce.item()
            dice_loss_sum += l_dice.item()
            num_batches += 1

            metrics_engine.update(preds, masks_gpu)

            # Generate qualitative visualizations for slices containing lesions
            if vis_saved < num_visualizations:
                b_preds = preds.cpu().numpy()
                b_masks = masks.numpy()
                b_imgs = images_norm.cpu().numpy()

                for b in range(len(b_preds)):
                    if vis_saved >= num_visualizations:
                        break

                    slice_global_idx = batch_idx * batch_size + b
                    if slice_global_idx >= len(df):
                        continue

                    row = df.iloc[slice_global_idx]
                    has_lesion = bool(row.get("has_tumor", False) or row.get("has_cyst", False) or row.get("has_kidney", False))

                    if has_lesion or np.sum(b_preds[b] > 0) > 100 or vis_saved < 2:
                        case_id = row.get("case_id", f"case_{slice_global_idx}")
                        slice_idx = row.get("slice_index", slice_global_idx)
                        img_slice = b_imgs[b, 0]
                        gt_slice = b_masks[b]
                        pr_slice = b_preds[b]

                        save_p = vis_dir / f"pred_{vis_saved + 1:02d}_{case_id}_slice_{int(slice_idx):03d}.png"
                        generate_qualitative_figure(img_slice, gt_slice, pr_slice, str(case_id), int(slice_idx), save_p)
                        vis_saved += 1

            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "DiceLoss": f"{l_dice.item():.3f}"
            })

    total_eval_time = max(0.001, time.time() - t_start)
    results = metrics_engine.compute()

    slices_evaluated = min(num_batches * batch_size, len(dataset))
    results["eval_loss"] = total_loss_sum / max(1, num_batches)
    results["eval_ce_loss"] = ce_loss_sum / max(1, num_batches)
    results["eval_dice_loss"] = dice_loss_sum / max(1, num_batches)
    results["eval_time_sec"] = total_eval_time
    results["slices_evaluated"] = slices_evaluated
    results["throughput_slices_per_sec"] = slices_evaluated / total_eval_time

    return results


# ==============================================================================
# 8. TIMELINE & REPORTING GENERATION
# ==============================================================================

def print_terminal_summary(results: Dict[str, Any], meta: Dict[str, Any], target_label: str):
    """Outputs a clean, professional formatted evaluation summary table."""
    cm = np.array(results["Confusion_Matrix"])

    print("\n" + "=" * 80)
    print(f"               KiTS23 EVALUATION SUMMARY — {target_label.upper()}")
    print("=" * 80)
    print(f"Checkpoint Epoch        : {meta.get('epoch', 'N/A')}")
    print(f"Slices Evaluated        : {results['slices_evaluated']:,} slices ({results['eval_time_sec']:.1f}s, {results['throughput_slices_per_sec']:.1f} slices/sec)")
    print(f"Evaluation Loss         : {results['eval_loss']:.4f} (CE: {results['eval_ce_loss']:.4f} | SoftDice: {results['eval_dice_loss']:.4f})")
    print("-" * 80)
    print(f"[*] OVERALL PIXEL ACCURACY     : {results['Pixel_Accuracy'] * 100:.2f}%")
    print(f"[+] FOREGROUND MACRO DICE (DSC): {results['Macro_Dice_Foreground']:.4f}  (Kidney + Tumor + Cyst average)")
    print(f"[+] MEAN IoU (Jaccard)         : {results['Mean_IoU']:.4f}")
    print("-" * 80)

    # Per-Class Table
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

    # Pixel Confusion Matrix Table
    print("\n[PIXEL CONFUSION MATRIX (Rows: True Class | Columns: Predicted Class)]")
    header = f"{'True Class':<12} | " + " | ".join([f"{name[:8]:>8}" for name in CLASS_NAMES])
    print(header)
    print("-" * len(header))
    for idx, name in enumerate(CLASS_NAMES):
        row_str = f"{name:<12} | " + " | ".join([f"{cm[idx, j]:>8,}" for j in range(len(CLASS_NAMES))])
        print(row_str)
    print("=" * 80 + "\n")


def export_run_reports(
    results: Dict[str, Any],
    meta: Dict[str, Any],
    args: argparse.Namespace,
    checkpoint_path: Path,
    run_dir: Path,
    run_id: str,
    timestamp: str
):
    """Saves structured JSON, CSV, Markdown, and metadata files into the run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Structured JSON
    json_path = run_dir / "evaluation_summary.json"
    full_export = {
        "run_id": run_id,
        "timestamp": timestamp,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_metadata": meta,
        "cli_arguments": vars(args),
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
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_export, f, indent=2)

    # 2. Per-Class CSV
    csv_path = run_dir / "per_class_metrics.csv"
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
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # 3. Execution Run Metadata
    meta_path = run_dir / "run_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(full_export, f, indent=2)

    # 4. Human-Readable Clinical Markdown Report
    target_name = f"Patient Case: {args.patient_case}" if args.patient_case else f"Dataset Split: {args.split.upper()}"
    md_report_path = run_dir / "clinical_report.md"
    md_lines = [
        f"# Clinical Evaluation Report: {target_name}",
        f"**Run Identifier:** `{run_id}`  ",
        f"**Timestamp:** {timestamp}  ",
        f"**Checkpoint:** `{checkpoint_path.name}`  ",
        "",
        "## 1. Executive Summary",
        f"- **Foreground Macro Dice (KiTS Official):** **{results['Macro_Dice_Foreground']:.4f}**",
        f"- **Overall Pixel Accuracy:** **{results['Pixel_Accuracy'] * 100:.2f}%**",
        f"- **Mean IoU (Jaccard Index):** **{results['Mean_IoU']:.4f}**",
        f"- **Total Slices Evaluated:** {results['slices_evaluated']:,}",
        f"- **Inference Speed:** {results['throughput_slices_per_sec']:.1f} slices/sec ({results['eval_time_sec']:.2f}s total)",
        "",
        "## 2. Anatomical Per-Class Breakdown",
        "| Anatomical Class | Dice (DSC) | IoU (Jaccard) | Precision (PPV) | Recall (Sens.) | Specificity (TNR) |",
        "|---|---|---|---|---|---|",
    ]
    for c in CLASS_NAMES:
        md_lines.append(
            f"| **{c}** | {results['Dice'][f'Dice_{c}']:.4f} | {results['IoU'][f'IoU_{c}']:.4f} | "
            f"{results['Precision'][f'Precision_{c}']:.4f} | {results['Recall'][f'Recall_{c}']:.4f} | "
            f"{results['Specificity'][f'Specificity_{c}']:.4f} |"
        )

    md_lines.extend([
        "",
        "## 3. Post-Processing & Calibration Configuration",
        f"- **Inference Mode:** {'Calibrated & Post-Processed' if (getattr(args, 'calibrated', False) or getattr(args, 'anatomical_containment', False)) else 'Standard Argmax'}",
        f"- **Probability Thresholds:** Tumor = {getattr(args, 'tumor_thresh', 0.35):.2f} | Cyst = {getattr(args, 'cyst_thresh', 0.28):.2f} | Kidney = {getattr(args, 'kidney_thresh', 0.50):.2f}",
        f"- **Anatomical Envelope Containment:** {'ACTIVE (GPU-accelerated)' if getattr(args, 'anatomical_containment', False) else 'Disabled'}",
        f"- **Morphological CC Noise Filter:** {'ACTIVE' if (getattr(args, 'min_kidney_voxels', 0) > 0 or getattr(args, 'min_tumor_voxels', 0) > 0 or getattr(args, 'min_cyst_voxels', 0) > 0) else 'Disabled'}",
        "",
        "## 4. Visual Artifacts",
        "- `confusion_matrix.png` — Pixel confusion matrix heatmap",
        "- `metrics_barchart.png` — Multi-metric comparison bar chart",
        "- `visualizations/` — 4-panel axial CT slice prediction overlays",
        "",
        "---",
        "*Report generated automatically by KiTS23 Evaluation Engine.*"
    ])

    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


def update_evaluation_timeline(
    timeline_dir: Path,
    run_record: Dict[str, Any]
):
    """
    Maintains a cumulative CSV ledger and Markdown timeline of all runs across time.
    """
    timeline_dir.mkdir(parents=True, exist_ok=True)
    csv_path = timeline_dir / "timeline.csv"
    md_path = timeline_dir / "timeline.md"

    # 1. Update CSV
    df_new = pd.DataFrame([run_record])
    if csv_path.exists():
        try:
            df_existing = pd.read_csv(csv_path)
            df_existing = df_existing[df_existing["Run_ID"] != run_record["Run_ID"]]
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception:
            df_combined = df_new
    else:
        df_combined = df_new

    df_combined.to_csv(csv_path, index=False)

    # 2. Update Markdown Timeline
    try:
        md_lines = [
            "# KiTS23 Evaluation Timeline & History",
            "",
            f"- **Total Evaluations:** {len(df_combined)}",
            f"- **Last Updated:** {run_record.get('Timestamp', 'N/A')}",
            f"- **Timeline Directory:** `{timeline_dir.resolve()}`",
            "",
            "---",
            "",
            "## Chronological Timeline (Newest Runs First)",
            "",
            "| # | Date & Time | Run ID | Target | Checkpoint | Slices | Macro Dice (FG) | Kidney Dice | Tumor Dice | Cyst Dice | Mean IoU | Pixel Acc | Folder |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
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
            r_dir = str(row.get("Run_Folder", run_id))

            line = (
                f"| {idx + 1} | {ts} | `{run_id}` | {target} | `{ckpt_name}` | {slices} | "
                f"**{macro_d}** | {kidney_d} | {tumor_d} | {cyst_d} | {miou} | {acc} | [`{run_id}`]({r_dir}) |"
            )
            md_lines.append(line)

        md_lines.extend([
            "",
            "---",
            "### Run Folder Artifacts",
            "Each timeline run folder contains:",
            "- `clinical_report.md` : Comprehensive clinical markdown report",
            "- `evaluation_summary.json` : Complete machine-readable metrics",
            "- `per_class_metrics.csv` : Tabular per-class DSC, IoU, Precision, Recall, Specificity",
            "- `confusion_matrix.png` : Pixel-level 4x4 confusion matrix heatmap",
            "- `metrics_barchart.png` : Multi-class comparative bar chart",
            "- `visualizations/` : 4-panel axial CT prediction overlays",
            "- `run_metadata.json` : Execution environment and hardware specs",
            ""
        ])

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")
    except Exception as e:
        print(f"[TIMELINE WARNING] Could not update timeline.md: {e}")


def update_latest_mirror(run_dir: Path, timeline_dir: Path):
    """Mirrors the latest run inside `<timeline_dir>/latest` for fast navigation."""
    latest_dir = timeline_dir / "latest"
    try:
        if latest_dir.exists():
            shutil.rmtree(latest_dir, ignore_errors=True)
        shutil.copytree(run_dir, latest_dir)
    except Exception:
        pass


# ==============================================================================
# 9. CLI ARGUMENT PARSER & MAIN EXECUTION
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KiTS23 Multi-Class U-Net Evaluation & Clinical Reporting CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core Evaluation Targets
    parser.add_argument(
        "--split", type=str, default="test", choices=["test", "val", "train"],
        help="Dataset split to evaluate ('test', 'val', or 'train')"
    )
    parser.add_argument(
        "--patient-case", type=str, default=None,
        help="Optional: Evaluate all axial slices of a specific patient case (e.g. 'case_00110')"
    )

    # Model & Data Paths (Auto-discovered by default)
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained checkpoint (.pth). If not specified, automatically finds best available model."
    )
    parser.add_argument(
        "--data-root", type=str, default=r"D:\archive",
        help="Root directory of dataset containing 'images' and 'masks' folders"
    )
    parser.add_argument(
        "--csv-path", type=str, default=None,
        help="Direct path to dataset.csv (default: <data-root>/dataset.csv)"
    )

    # Performance & Execution Controls
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for parallel inference"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader worker processes"
    )
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="Limit number of batches to evaluate for quick test (e.g. 50 or 100)"
    )
    parser.add_argument(
        "--tta", action="store_true", default=False,
        help="Enable Test-Time Augmentation (horizontal flip averaging) for boosted Dice score"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference ('cuda' or 'cpu')"
    )
    parser.add_argument(
        "--cpu", action="store_true", default=False,
        help="Convenience flag to force CPU execution"
    )
    parser.add_argument(
        "--amp", action="store_true", default=True,
        help="Use Automatic Mixed Precision (AMP FP16) on GPU"
    )

    # Reporting & Timeline Directory
    parser.add_argument(
        "--output-dir", type=str, default="reports/timeline",
        help="Root timeline directory where timestamped run folders will be stored"
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Optional custom label/tag to append to the timestamped timeline folder"
    )
    parser.add_argument(
        "--num-visualizations", type=int, default=10,
        help="Number of 4-panel axial CT overlay figures to generate"
    )

    # Post-Processing & Calibrated Thresholding Controls
    parser.add_argument(
        "--calibrated", action="store_true", default=False,
        help="Enable calibrated probability thresholding for boosted lesion sensitivity"
    )
    parser.add_argument(
        "--tumor-thresh", type=float, default=0.35,
        help="Calibrated probability threshold for Tumor (default: 0.35 to recover missed tumor)"
    )
    parser.add_argument(
        "--cyst-thresh", type=float, default=0.28,
        help="Calibrated probability threshold for Cyst (default: 0.28 to resolve kidney-cyst confusion)"
    )
    parser.add_argument(
        "--kidney-thresh", type=float, default=0.50,
        help="Probability threshold for Kidney parenchyma (default: 0.50)"
    )
    parser.add_argument(
        "--anatomical-containment", action="store_true", default=False,
        help="Enable GPU anatomical envelope containment to suppress out-of-kidney false positive noise"
    )
    parser.add_argument(
        "--min-tumor-voxels", type=int, default=0,
        help="Filter out isolated tumor clusters smaller than N voxels (e.g. 20)"
    )
    parser.add_argument(
        "--min-cyst-voxels", type=int, default=0,
        help="Filter out isolated cyst clusters smaller than N voxels (e.g. 10)"
    )
    parser.add_argument(
        "--min-kidney-voxels", type=int, default=0,
        help="Filter out isolated kidney clusters smaller than N voxels (e.g. 30)"
    )
    parser.add_argument(
        "--post-process", action="store_true", default=False,
        help="Comprehensive post-processing: calibrated thresholding + anatomical containment + CC noise filter"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.patient_case and not args.patient_case.startswith("case_"):
        args.patient_case = f"case_{args.patient_case}"

    # Auto-expand --post-process shortcut
    if args.post_process:
        args.calibrated = True
        args.anatomical_containment = True
        if args.min_tumor_voxels == 0:
            args.min_tumor_voxels = 20
        if args.min_cyst_voxels == 0:
            args.min_cyst_voxels = 10
        if args.min_kidney_voxels == 0:
            args.min_kidney_voxels = 30

    # Hardware Device Resolution
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # 1. Resolve Dataset and Metadata
    data_root, csv_path = auto_detect_dataset(args.data_root, args.csv_path)
    image_dir = data_root / "images"
    mask_dir = data_root / "masks"

    df_meta = pd.read_csv(csv_path)
    if args.patient_case:
        split_df = df_meta[df_meta["case_id"] == args.patient_case].sort_values("slice_index").reset_index(drop=True)
        if len(split_df) == 0:
            print(f"[ERROR] Patient case '{args.patient_case}' not found in {csv_path}!")
            sys.exit(1)
        target_label = args.patient_case
        target_display = f"Patient {args.patient_case}"
    else:
        split_df = df_meta[df_meta["split"] == args.split].copy().reset_index(drop=True)
        if len(split_df) == 0:
            print(f"[ERROR] Split '{args.split}' contains 0 records in {csv_path}!")
            sys.exit(1)
        target_label = f"{args.split}"
        target_display = f"Split {args.split.upper()}"

    dataset = KiTS23Dataset(split_df, image_dir, mask_dir)

    # 2. Timeline Folder Resolution
    timeline_root = Path(args.output_dir)
    now = datetime.now()
    timestamp_folder = now.strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_name:
        clean_tag = "".join(c if (c.isalnum() or c in "_-") else "_" for c in args.run_name)
        run_folder_name = f"{timestamp_folder}_{target_label}_{clean_tag}"
    else:
        run_folder_name = f"{timestamp_folder}_{target_label}"

    run_dir = timeline_root / run_folder_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("                    KiTS23 MODEL EVALUATION CLI")
    print("=" * 80)
    print(f"Target Evaluation Split : {target_display} ({len(split_df):,} total axial slices)")
    print(f"Dataset Root            : {data_root}")
    print(f"Timeline Folder (Output): {run_dir.resolve()}")
    print(f"Device Acceleration     : {device.type.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print("=" * 80)

    # 3. Model Checkpoint Resolution & Loading
    checkpoint_path = find_best_checkpoint(args.checkpoint)
    model, meta = load_model_for_evaluation(checkpoint_path, device)

    # 4. Execute High-Throughput Inference
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
        tta=args.tta,
        calibrated=args.calibrated,
        tumor_thresh=args.tumor_thresh,
        cyst_thresh=args.cyst_thresh,
        kidney_thresh=args.kidney_thresh,
        anatomical_containment=args.anatomical_containment,
        min_kidney_voxels=args.min_kidney_voxels,
        min_tumor_voxels=args.min_tumor_voxels,
        min_cyst_voxels=args.min_cyst_voxels
    )

    # 5. Visual Artifacts Generation
    cm_plot_path = run_dir / "confusion_matrix.png"
    plot_confusion_matrix(np.array(results["Confusion_Matrix"]), CLASS_NAMES, cm_plot_path)

    bar_plot_path = run_dir / "metrics_barchart.png"
    plot_metrics_barchart(results, CLASS_NAMES, bar_plot_path)

    # 6. Export Reports & Metadata
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    export_run_reports(
        results=results,
        meta=meta,
        args=args,
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        run_id=run_folder_name,
        timestamp=timestamp_str
    )

    # 7. Update Cumulative Timeline Ledger & Latest Mirror
    run_record = {
        "Timestamp": timestamp_str,
        "Run_ID": run_folder_name,
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
        "Run_Folder": run_folder_name
    }
    update_evaluation_timeline(timeline_root, run_record)
    update_latest_mirror(run_dir, timeline_root)

    # 8. Print Console Report
    print_terminal_summary(results, meta, target_display)

    print("=" * 80)
    print(f"[SUCCESS] Evaluation completed successfully!")
    print(f"  • Timeline Run Folder : {run_dir.resolve()}")
    print(f"  • Clinical Report     : {(run_dir / 'clinical_report.md').resolve()}")
    print(f"  • Summary Metrics     : {(run_dir / 'evaluation_summary.json').resolve()}")
    print(f"  • Timeline Index      : {(timeline_root / 'timeline.md').resolve()}")
    print(f"  • Latest Mirror       : {(timeline_root / 'latest').resolve()}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
