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
import numpy as np

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


# Global singleton instances
_model: Optional[ResidualUNet] = None
_device: Optional[torch.device] = None
_checkpoint_path: Optional[str] = None
_onnx_session: Optional[Any] = None
_onnx_path: Optional[str] = None
_configured_cpu_threads: Optional[int] = None


def is_model_loaded() -> bool:
    """Returns whether a checkpoint-backed model is currently available."""
    return _model is not None


def reset_singleton_state() -> None:
    """Resets global singleton instances (primarily for testing and re-configuration)."""
    global _model, _device, _checkpoint_path, _onnx_session, _onnx_path, _configured_cpu_threads
    _model = None
    _device = None
    _checkpoint_path = None
    _onnx_session = None
    _onnx_path = None
    _configured_cpu_threads = None


def get_cpu_threads() -> int:
    """Returns validated CPU thread count for PyTorch and ONNX Runtime.

    Configurable via set_cpu_threads() or CPU_NUM_THREADS/OMP_NUM_THREADS environment variables.
    Defaults to min(cpu_count, 8) to prevent OpenMP thread oversubscription.
    """
    global _configured_cpu_threads
    if _configured_cpu_threads is not None and _configured_cpu_threads > 0:
        return _configured_cpu_threads

    env_threads = os.getenv("CPU_NUM_THREADS") or os.getenv("OMP_NUM_THREADS")
    if env_threads:
        try:
            val = int(env_threads)
            if val > 0:
                return val
        except ValueError:
            pass
    cpu_count = os.cpu_count() or 4
    return max(1, min(cpu_count, 8))


def set_cpu_threads(n_threads: int) -> None:
    """Sets CPU thread count for PyTorch and invalidates cached ONNX session."""
    global _onnx_session, _configured_cpu_threads
    if n_threads > 0:
        _configured_cpu_threads = n_threads
        torch.set_num_threads(n_threads)
        _onnx_session = None


def get_device() -> torch.device:
    """Returns CUDA device if available and functional, otherwise gracefully falls back to CPU.

    - Automatically detects NVIDIA GPU with CUDA support.
    - Tests tensor allocation to ensure driver and memory are operational.
    - Gracefully falls back to CPU without crashing when CUDA/NVIDIA drivers are missing.
    - Configures CPU thread optimization when operating in CPU mode.
    - Never requires manual user intervention.
    """
    global _device
    if _device is None:
        if torch.cuda.is_available():
            try:
                # Test allocation on GPU to ensure driver and VRAM are fully operational
                test_tensor = torch.zeros(1, device="cuda")
                _device = torch.device("cuda")
                # cuDNN benchmark mode for fixed 512x512 inputs and TF32 acceleration
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                gpu_name = torch.cuda.get_device_name(0)
                print(f"[DEVICE] Hardware acceleration ACTIVE: GPU ({gpu_name})")
            except Exception as e:
                print(f"[DEVICE WARNING] CUDA is detected but failed to initialize ({e}). Falling back to CPU mode.")
                _device = torch.device("cpu")
        else:
            print("[DEVICE] No CUDA GPU detected. Running in standard CPU mode.")
            _device = torch.device("cpu")

        if _device.type == "cpu":
            threads = get_cpu_threads()
            torch.set_num_threads(threads)
            print(f"[DEVICE] Configured PyTorch CPU thread count: {threads}")
    return _device


def get_available_backend() -> str:
    """Returns the active inference backend: 'pytorch_cuda', 'onnx_cpu', or 'pytorch_cpu'.

    Can be explicitly configured via INFERENCE_BACKEND environment variable ('auto', 'pytorch_cuda',
    'onnx_cpu', 'pytorch_cpu'). Defaults to 'auto'.
    """
    configured = os.getenv("INFERENCE_BACKEND", "auto").strip().lower()
    device = get_device()

    if configured in ("pytorch_cuda", "cuda") and device.type == "cuda":
        return "pytorch_cuda"
    elif configured == "onnx_cpu":
        return "onnx_cpu"
    elif configured in ("pytorch_cpu", "cpu"):
        return "pytorch_cpu"

    # Automatic selection based on detected hardware:
    if device.type == "cuda":
        return "pytorch_cuda"

    # CPU mode: check if onnxruntime is available and functional
    try:
        import onnxruntime
        return "onnx_cpu"
    except ImportError:
        return "pytorch_cpu"


def export_and_validate_onnx(
    model: Optional[ResidualUNet] = None,
    checkpoint_path: Optional[str] = None,
    onnx_path: Optional[str] = None,
    force_reexport: bool = False,
    tolerance: float = 1e-3,
    num_threads: Optional[int] = None
) -> str:
    """Exports trained PyTorch ResidualUNet model to ONNX format with dynamic batch axis and validates parity."""
    if onnx_path is None:
        if checkpoint_path is None:
            checkpoint_path = os.getenv("MODEL_CHECKPOINT", str(Path("notebookV3/checkpoints/best_unet_kits23.pth").resolve()))
        ckpt_p = Path(checkpoint_path)
        onnx_path = str(ckpt_p.with_suffix(".onnx"))

    onnx_file = Path(onnx_path)
    if onnx_file.exists() and not force_reexport:
        return str(onnx_file.resolve())

    onnx_file.parent.mkdir(parents=True, exist_ok=True)

    if model is None:
        model, _ = load_model(checkpoint_path)

    # Export on CPU in eval mode to produce pure portable computation graph
    model_cpu = model.to("cpu")
    model_cpu.eval()

    dummy_input = torch.zeros(1, 1, 512, 512, dtype=torch.float32)

    print(f"[ONNX EXPORT] Exporting Residual U-Net to ONNX format: {onnx_file}...")
    torch.onnx.export(
        model_cpu,
        dummy_input,
        str(onnx_file),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,
    )

    # Validate ONNX model structural integrity
    import onnx
    onnx_model = onnx.load(str(onnx_file))
    onnx.checker.check_model(onnx_model)
    print("[ONNX EXPORT] Model structure check passed.")

    # Validate numerical outputs against PyTorch
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = num_threads or get_cpu_threads()
    session = ort.InferenceSession(str(onnx_file), sess_options=opts, providers=["CPUExecutionProvider"])

    test_input = np.random.RandomState(42).randn(1, 1, 512, 512).astype(np.float32)
    with torch.inference_mode():
        pt_logits = model_cpu(torch.from_numpy(test_input)).numpy()
    ort_logits = session.run(["output"], {"input": test_input})[0]

    max_diff = float(np.max(np.abs(pt_logits - ort_logits)))
    pt_pred = np.argmax(pt_logits, axis=1)
    ort_pred = np.argmax(ort_logits, axis=1)
    mask_mismatch = int(np.sum(pt_pred != ort_pred))

    print(f"[ONNX VALIDATION] Max logit difference: {max_diff:.6f} (tolerance={tolerance}), mask mismatch: {mask_mismatch} pixels.")
    if max_diff > tolerance:
        print(f"[ONNX WARNING] Logit difference {max_diff:.6f} exceeds tolerance {tolerance}.")

    return str(onnx_file.resolve())


def get_onnx_session(
    onnx_path: Optional[str] = None,
    num_threads: Optional[int] = None,
    force_reload: bool = False
) -> Optional[Any]:
    """Returns cached ONNX Runtime InferenceSession with CPUExecutionProvider and thread configuration."""
    global _onnx_session, _onnx_path
    if _onnx_session is not None and not force_reload:
        return _onnx_session

    try:
        import onnxruntime as ort
    except ImportError:
        print("[ONNX WARNING] onnxruntime is not installed. CPU inference will use PyTorch fallback.")
        return None

    if onnx_path is None:
        checkpoint_path = os.getenv("MODEL_CHECKPOINT", str(Path("notebookV3/checkpoints/best_unet_kits23.pth").resolve()))
        onnx_path = str(Path(checkpoint_path).with_suffix(".onnx"))

    if not os.path.exists(onnx_path):
        try:
            onnx_path = export_and_validate_onnx(onnx_path=onnx_path, num_threads=num_threads)
        except Exception as e:
            print(f"[ONNX WARNING] ONNX export failed ({e}). Falling back to PyTorch CPU.")
            return None

    threads = num_threads or get_cpu_threads()
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    try:
        session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        _onnx_session = session
        _onnx_path = onnx_path
        print(f"[ONNX] ONNX Runtime CPU session initialized successfully (threads={threads}).")
        return _onnx_session
    except Exception as e:
        print(f"[ONNX WARNING] Failed to initialize ONNX Runtime session ({e}). Falling back to PyTorch CPU.")
        return None


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
        # If loading onto GPU fails (e.g. CUDA OOM or driver issue), gracefully fall back to CPU
        if device.type == "cuda":
            print(f"[LOAD WARNING] GPU placement failed ({e}). Automatically falling back to CPU...")
            device = torch.device("cpu")
            _device = device
            threads = get_cpu_threads()
            torch.set_num_threads(threads)
            model = ResidualUNet(in_channels=1, num_classes=4, base_filters=32).to(device)
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
            model.load_state_dict(state_dict)
        else:
            raise e

    model.eval()
    _model = model
    return _model, device


