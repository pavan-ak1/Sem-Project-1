import os
import sys
import platform
import pathlib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# Cross-platform compatibility fix:
# Checkpoints saved on Windows contain `pathlib.WindowsPath` in pickled metadata
# (e.g. ckpt['args']['data_root']), which raises:
# `NotImplementedError: cannot instantiate 'WindowsPath' on your system`
# when deserialized on Linux/Docker (and vice-versa for PosixPath on Windows).
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

import torch
import torch.nn as nn
import torch.nn.functional as F

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


# Global singleton instance
_model: Optional[ResidualUNet] = None
_device: Optional[torch.device] = None
_checkpoint_path: Optional[str] = None

def is_model_loaded() -> bool:
    """Returns whether a checkpoint-backed model is currently available."""
    return _model is not None


def get_device() -> torch.device:
    """Returns CUDA device if available and functional, otherwise falls back to CPU."""
    global _device
    if _device is None:
        if torch.cuda.is_available():
            try:
                # Test allocation on GPU to ensure driver and memory are operational
                test_tensor = torch.zeros(1, device="cuda")
                _device = torch.device("cuda")
                gpu_name = torch.cuda.get_device_name(0)
                print(f"[DEVICE] Hardware acceleration ACTIVE: GPU ({gpu_name})")
            except Exception as e:
                print(f"[DEVICE WARNING] CUDA is detected but failed to initialize ({e}). Falling back to CPU mode.")
                _device = torch.device("cpu")
        else:
            print("[DEVICE] No CUDA GPU detected. Running in standard CPU mode.")
            _device = torch.device("cpu")

        if _device.type == "cpu":
            # Prevent OpenMP thread oversubscription on CPU
            cpu_count = os.cpu_count() or 4
            torch.set_num_threads(min(cpu_count, 8))
    return _device


def load_model(checkpoint_path: Optional[str] = None) -> Tuple[ResidualUNet, torch.device]:
    """Loads ResidualUNet model with automatic GPU/CPU placement and fallback."""
    global _model, _checkpoint_path, _device
    if _model is not None:
        return _model, get_device()

    device = get_device()
    if checkpoint_path is None:
        checkpoint_path = os.getenv("MODEL_CHECKPOINT", str(Path("notebookV3/checkpoints/best_unet_kits23.pth").resolve()))

    _checkpoint_path = checkpoint_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    try:
        model = ResidualUNet(in_channels=1, num_classes=4, base_filters=32).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
        model.load_state_dict(state_dict)
    except Exception as e:
        # If loading onto GPU fails (e.g. CUDA OOM), gracefully fall back to CPU
        if device.type == "cuda":
            print(f"[LOAD WARNING] GPU placement failed ({e}). Automatically falling back to CPU...")
            device = torch.device("cpu")
            _device = device
            model = ResidualUNet(in_channels=1, num_classes=4, base_filters=32).to(device)
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
            model.load_state_dict(state_dict)
        else:
            raise e

    model.eval()
    _model = model
    return _model, device

