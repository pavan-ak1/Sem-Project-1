#!/usr/bin/env python3
"""
================================================================================
KiTS23 Retraining Engine with Class-Weighted Focal & Asymmetric Tversky Loss
================================================================================

This retraining engine directly targets the two major defects identified in
the KiTS23 confusion matrix:
  1. Tumor Under-Segmentation (31.2% missed as Background)
  2. Cyst/Kidney Confusion (21.5% absorbed into Kidney parenchyma)

Key Loss Formulations:
  - Class-Weighted Focal Loss (gamma=2.0, weights: BG=0.1, Kidney=1.0, Tumor=3.0, Cyst=6.0):
      Dynamically downweights easy background pixels (99.17% of total pixels)
      and amplifies hard lesion boundaries.
  - Asymmetric Multi-Class Tversky Loss (alpha=0.3, beta=0.7, ignore_bg=True):
      Penalizes False Negatives (beta=0.7) more severely than False Positives (alpha=0.3),
      directly driving tumor and cyst sensitivity/recall upward.

Checkpoint Output:
  - Best Model   : notebookV3/checkpoints/best_model_retrain_tversky_loss.pth
  - Latest State : notebookV3/checkpoints/latest_checkpoint_tversky_loss.pth
================================================================================
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

# Safeguard Windows console text encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import v2
from torchvision import tv_tensors

from backend.model_loader import ResidualUNet
from evaluate import standardize_spatial_dims, ClinicalEvaluationMetrics, CLASS_NAMES


# ==============================================================================
# 1. CLASS-WEIGHTED FOCAL & ASYMMETRIC TVERSKY LOSS FUNCTIONS
# ==============================================================================

class ClassWeightedFocalLoss(nn.Module):
    """
    Multi-Class Focal Loss with Class-Specific Alpha Weighting.
    Addresses extreme foreground-background imbalance by dynamically
    down-weighting easy background pixels with focusing parameter gamma=2.0.
    """
    def __init__(
        self,
        num_classes: int = 4,
        class_weights: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        # Default weights: Background=0.1, Kidney=1.0, Tumor=3.0, Cyst=6.0
        if class_weights is None:
            self.class_weights = torch.tensor([0.1, 1.0, 3.0, 6.0], dtype=torch.float32)
        else:
            self.class_weights = class_weights.float()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device = logits.device
        weights = self.class_weights.to(device)

        ce_loss = F.cross_entropy(logits, targets, reduction="none")  # (B, H, W)
        p_t = torch.exp(-ce_loss)  # (B, H, W)
        focal_factor = (1.0 - p_t) ** self.gamma
        alpha_t = weights[targets]

        focal_loss = alpha_t * focal_factor * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class MultiClassTverskyLoss(nn.Module):
    """
    Multi-Class Tversky Loss with False-Negative Penalty.
    Setting beta=0.7 (FN penalty) > alpha=0.3 (FP penalty) forces the network
    to penalize missed tumor and cyst regions more heavily than boundary expansion.
    """
    def __init__(
        self,
        num_classes: int = 4,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-5,
        ignore_bg: bool = True,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # Penalty on False Positives
        self.beta = beta    # Penalty on False Negatives (higher beta = higher recall)
        self.smooth = smooth
        self.ignore_bg = ignore_bg
        # Prioritize rare pathologies: Kidney=1.0, Tumor=2.0, Cyst=3.0
        if class_weights is None:
            self.class_weights = torch.tensor([0.0, 1.0, 2.0, 3.0] if ignore_bg else [0.2, 1.0, 2.0, 3.0], dtype=torch.float32)
        else:
            self.class_weights = class_weights.float()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).contiguous().float()

        probs_flat = probs.view(probs.size(0), self.num_classes, -1)
        targets_flat = targets_one_hot.view(targets_one_hot.size(0), self.num_classes, -1)

        tp = (probs_flat * targets_flat).sum(dim=2)
        fp = (probs_flat * (1.0 - targets_flat)).sum(dim=2)
        fn = ((1.0 - probs_flat) * targets_flat).sum(dim=2)

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        tversky_loss_per_class = 1.0 - tversky_index  # (B, C)

        weights = self.class_weights.to(logits.device)
        if self.ignore_bg:
            fg_loss = tversky_loss_per_class[:, 1:]
            fg_weights = weights[1:]
            weighted_loss = (fg_loss * fg_weights).sum(dim=1) / fg_weights.sum()
            return weighted_loss.mean()
        else:
            weighted_loss = (tversky_loss_per_class * weights).sum(dim=1) / weights.sum()
            return weighted_loss.mean()


class CompositeFocalTverskyLoss(nn.Module):
    """
    Combined Class-Weighted Focal Loss + Asymmetric Tversky Loss.
    """
    def __init__(
        self,
        num_classes: int = 4,
        focal_weight: float = 1.0,
        tversky_weight: float = 1.0,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        ignore_bg_in_tversky: bool = True
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.focal_loss = ClassWeightedFocalLoss(
            num_classes=num_classes,
            class_weights=torch.tensor([0.1, 1.0, 3.0, 6.0]),
            gamma=focal_gamma
        )
        self.tversky_loss = MultiClassTverskyLoss(
            num_classes=num_classes,
            alpha=tversky_alpha,
            beta=tversky_beta,
            ignore_bg=ignore_bg_in_tversky
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l_focal = self.focal_loss(logits, targets)
        l_tversky = self.tversky_loss(logits, targets)
        l_total = (self.focal_weight * l_focal) + (self.tversky_weight * l_tversky)
        return l_total, l_focal, l_tversky


# ==============================================================================
# 2. FAST DATASET & GPU AUGMENTATION
# ==============================================================================

class FastKiTS23Dataset(Dataset):
    """Passes compact integer arrays to avoid CPU float bloat and IPC stalls."""
    def __init__(self, df: pd.DataFrame, image_dir: Path, mask_dir: Path):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.filenames = self.df["filename"].to_numpy()
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
        return torch.from_numpy(img_np).unsqueeze(0), torch.from_numpy(msk_np)


class GPUBatchAugmentation(nn.Module):
    """CUDA Tensor Core batch augmentation running in <4ms."""
    def __init__(self):
        super().__init__()
        self.transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.2),
            v2.RandomAffine(
                degrees=15,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                interpolation=v2.InterpolationMode.BILINEAR
            ),
            v2.RandomApply([v2.ColorJitter(brightness=0.1, contrast=0.1)], p=0.3),
        ])

    @torch.no_grad()
    def forward(self, images: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        img_tv = tv_tensors.Image(images)
        msk_tv = tv_tensors.Mask(masks)
        img_aug, msk_aug = self.transforms(img_tv, msk_tv)
        img_out = torch.clamp(img_aug.as_subclass(torch.Tensor), 0.0, 1.0)
        msk_out = msk_aug.as_subclass(torch.Tensor)
        return img_out, msk_out


def compute_slice_sampling_weights(df: pd.DataFrame) -> torch.Tensor:
    """Computes per-slice sampling weights prioritizing rare tumor & cyst pathologies."""
    weights = np.zeros(len(df), dtype=np.float32)
    has_c = df["has_cyst"].to_numpy(dtype=bool)
    has_t = df["has_tumor"].to_numpy(dtype=bool)
    has_k = df["has_kidney"].to_numpy(dtype=bool)
    is_bg = ~(has_c | has_t | has_k)

    weights[is_bg] = 0.5
    weights[has_k & ~(has_t | has_c)] = 3.0
    weights[has_t & ~has_c] = 10.0      # Boosted tumor sampling
    weights[has_c] = 20.0               # Extra boosted cyst sampling (rarest pathology)
    return torch.from_numpy(weights).double()


# ==============================================================================
# 3. CHECKPOINT SERIALIZATION HELPERS
# ==============================================================================

def save_checkpoint(
    filepath: Path,
    epoch: int,
    batch_idx: int,
    epoch_completed: bool,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    best_dice: float,
    history: Dict[str, List]
):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "epoch_completed": epoch_completed,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_macro_dice": best_dice,
        "history": history,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "loss_type": "CompositeFocalTverskyLoss",
        "description": "Retrained with Class-Weighted Focal (gamma=2.0) + Asymmetric Tversky (beta=0.7)"
    }
    torch.save(state, filepath)


# ==============================================================================
# 4. MAIN RETRAINING EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="KiTS23 Retraining with Class-Weighted Focal / Tversky Loss")
    parser.add_argument("--epochs", type=int, default=10, help="Total retraining epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4 for fine-tuning)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--warm-start", type=str, default="notebookV3/checkpoints/best_unet_kits23.pth",
                        help="Path to initial weights to warm-start/fine-tune from")
    parser.add_argument("--scratch", action="store_true", default=False, help="Train from random scratch weights")
    parser.add_argument("--data-root", type=str, default=r"D:\archive", help="Dataset directory")
    parser.add_argument("--val-batches", type=int, default=200, help="Max validation batches to evaluate per epoch")
    parser.add_argument("--save-interval", type=int, default=100, help="Batch save interval for latest checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 80)
    print("      KiTS23 RETRAINING: CLASS-WEIGHTED FOCAL + ASYMMETRIC TVERSKY LOSS")
    print("=" * 80)
    print(f"Device Acceleration    : {device.type.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"Target Total Epochs    : {args.epochs}")
    print(f"Initial Learning Rate  : {args.lr:.1e}")
    print(f"Batch Size             : {args.batch_size}")
    print(f"Loss Formulation       : 1.0 * Focal (gamma=2, weights=[0.1, 1, 3, 6]) + 1.0 * Tversky (alpha=0.3, beta=0.7)")
    print("=" * 80)

    # 1. Dataset Resolution
    data_root = Path(args.data_root)
    csv_path = data_root / "dataset.csv"
    image_dir = data_root / "images"
    mask_dir = data_root / "masks"

    if not csv_path.exists():
        raise FileNotFoundError(f"dataset.csv not found at: {csv_path.resolve()}")

    df_all = pd.read_csv(csv_path)
    train_df = df_all[df_all["split"] == "train"].reset_index(drop=True)
    val_df = df_all[df_all["split"] == "val"].reset_index(drop=True)

    print(f"Training Slices        : {len(train_df):,}")
    print(f"Validation Slices      : {len(val_df):,}")

    train_ds = FastKiTS23Dataset(train_df, image_dir, mask_dir)
    val_ds = FastKiTS23Dataset(val_df, image_dir, mask_dir)

    train_sampler = WeightedRandomSampler(
        weights=compute_slice_sampling_weights(train_df),
        num_samples=len(train_df),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=0, pin_memory=(device.type == "cuda"), drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"), drop_last=False
    )

    target_batches = len(train_loader)

    # 2. Model & Checkpoints Setup
    CHECKPOINT_DIR = Path("notebookV3/checkpoints")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_CKPT_PATH = CHECKPOINT_DIR / "best_model_retrain_tversky_loss.pth"
    LATEST_CKPT_PATH = CHECKPOINT_DIR / "latest_checkpoint_tversky_loss.pth"

    model = ResidualUNet(in_channels=1, num_classes=4, base_filters=32).to(device)

    # Warm-start weights
    if not args.scratch and args.warm_start and Path(args.warm_start).exists():
        print(f"\n[WARM-START] Loading base weights from: {args.warm_start}")
        ckpt = torch.load(args.warm_start, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
        model.load_state_dict(sd)
        print("[WARM-START] Base weights loaded successfully! Initiating fine-tuning.")
    else:
        print("\n[TRAINING] Initializing model with fresh random weights.")

    # 3. Loss, Optimizer, Scheduler & Scaler
    criterion = CompositeFocalTverskyLoss(
        num_classes=4,
        focal_weight=1.0,
        tversky_weight=1.0,
        focal_gamma=2.0,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        ignore_bg_in_tversky=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gpu_augmenter = GPUBatchAugmentation().to(device)
    metrics_engine = ClinicalEvaluationMetrics(num_classes=4, class_names=CLASS_NAMES)

    best_macro_dice = 0.0
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "lr": []}

    print("\n" + "=" * 80)
    print(f"STARTING RETRAINING PIPELINE: 1 TO {args.epochs} EPOCHS")
    print(f"Target Checkpoint: {BEST_CKPT_PATH.resolve()}")
    print("=" * 80 + "\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        model.train()
        train_loss_sum = 0.0
        focal_loss_sum = 0.0
        tversky_loss_sum = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{args.epochs:02d}]", unit="batch", dynamic_ncols=True)

        for batch_idx, (images, masks) in enumerate(pbar):
            images = (images.to(device, non_blocking=True).float() / 65535.0)
            masks = masks.to(device, dtype=torch.long, non_blocking=True)

            images, masks = gpu_augmenter(images, masks)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss, l_focal, l_tversky = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            focal_loss_sum += l_focal.item()
            tversky_loss_sum += l_tversky.item()
            num_batches += 1

            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Focal": f"{l_focal.item():.3f}",
                "Tversky": f"{l_tversky.item():.3f}",
                "LR": f"{current_lr:.2e}"
            })

            if (batch_idx + 1) % args.save_interval == 0:
                save_checkpoint(
                    LATEST_CKPT_PATH, epoch=epoch, batch_idx=batch_idx, epoch_completed=False,
                    model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    best_dice=best_macro_dice, history=history
                )

        train_loss_avg = train_loss_sum / max(1, num_batches)

        # Validation
        print(f"\n>>> Epoch {epoch} complete ({time.time() - t0:.1f}s). Evaluating validation cohort...")
        model.eval()
        metrics_engine.reset()
        val_loss_sum = 0.0
        val_batches_count = 0

        with torch.no_grad():
            for v_idx, (v_imgs, v_msks) in enumerate(val_loader):
                if v_idx >= args.val_batches:
                    break
                v_imgs = (v_imgs.to(device, non_blocking=True).float() / 65535.0)
                v_msks = v_msks.to(device, dtype=torch.long, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    v_logits = model(v_imgs)
                    v_loss, _, _ = criterion(v_logits, v_msks)

                val_loss_sum += v_loss.item()
                val_batches_count += 1
                v_preds = torch.argmax(v_logits, dim=1)
                metrics_engine.update(v_preds, v_msks)

        val_res = metrics_engine.compute()
        val_loss_avg = val_loss_sum / max(1, val_batches_count)
        macro_dice = val_res["Macro_Dice_Foreground"]
        is_best = macro_dice > best_macro_dice

        history["train_loss"].append(train_loss_avg)
        history["val_loss"].append(val_loss_avg)
        history["val_dice"].append(macro_dice)
        history["lr"].append(current_lr)

        print(f"  Validation Loss     : {val_loss_avg:.4f}")
        print(f"  Macro Dice (FG)     : {macro_dice:.4f} (Kidney: {val_res['Dice']['Dice_Kidney']:.4f} | Tumor: {val_res['Dice']['Dice_Tumor']:.4f} | Cyst: {val_res['Dice']['Dice_Cyst']:.4f})")
        print(f"  Mean IoU            : {val_res['Mean_IoU']:.4f} | Pixel Acc: {val_res['Pixel_Accuracy']*100:.2f}%")

        if is_best:
            best_macro_dice = macro_dice
            save_checkpoint(
                BEST_CKPT_PATH, epoch=epoch, batch_idx=target_batches - 1, epoch_completed=True,
                model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                best_dice=best_macro_dice, history=history
            )
            print(f"  🏆 [NEW BEST MODEL] Saved to: {BEST_CKPT_PATH.resolve()} (Macro Dice: {macro_dice:.4f})\n")

        save_checkpoint(
            LATEST_CKPT_PATH, epoch=epoch, batch_idx=target_batches - 1, epoch_completed=True,
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            best_dice=best_macro_dice, history=history
        )

        scheduler.step()

    print("\n" + "=" * 80)
    print("RETRAINING PROCESS COMPLETE!")
    print(f"Best Macro Dice : {best_macro_dice:.4f}")
    print(f"Best Checkpoint : {BEST_CKPT_PATH.resolve()}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
