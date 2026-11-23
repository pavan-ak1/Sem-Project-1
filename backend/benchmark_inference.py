"""Reproducible GPU & CPU inference benchmark for the 2D kidney CT pipeline.

Examples:
    # Auto-detect hardware and benchmark (GPU or CPU)
    python -m backend.benchmark_inference --case-id case_00110

    # CPU comparison: PyTorch CPU vs ONNX Runtime CPU across multiple thread counts
    python -m backend.benchmark_inference --case-id case_00110 --threads 2 4 8 --batch-sizes 1 4

    # Run on synthetic volume (useful if KiTS23 dataset archive is not locally mounted)
    python -m backend.benchmark_inference --synthetic --num-slices 16 --threads 4

Measures:
    - Selected execution device (CUDA vs CPU)
    - Selected execution backend (PyTorch CUDA, ONNX Runtime CPU, PyTorch CPU)
    - Model loading time (PyTorch weights deserialization vs ONNX session initialization)
    - Average and median inference times (per slice and per volume)
    - Peak memory usage (GPU VRAM via torch.cuda and CPU RSS RAM via psutil)
    - Total prediction time including preprocessing, 3D CCA postprocessing, and mask export
    - Numerical consistency and segmentation metrics (Dice / IoU)
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import psutil
import torch

from backend.inference_engine import (
    CSV_PATH,
    IMAGE_DIR,
    MASK_DIR,
    clean_segmentation_volume,
    filter_prediction_slices,
    predict_tensor,
    predict_volume_batched,
    standardize_dims,
)
from backend.model_loader import (
    load_model,
    get_device,
    get_available_backend,
    get_onnx_session,
    get_cpu_threads,
    set_cpu_threads,
    export_and_validate_onnx,
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def gpu_utilization() -> Optional[float]:
    """Read instantaneous GPU utilization when nvidia-smi is available."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return float(output.strip().splitlines()[0])
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def get_current_rss_mb() -> float:
    """Returns current process Resident Set Size (RSS) memory in MiB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)


def load_case(case_id: str) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    """Load the same 512 x 512 normalized slices used by production inference."""
    start = time.perf_counter()
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset index not found at {CSV_PATH}.")

    metadata = pd.read_csv(CSV_PATH)
    case_df = metadata[metadata["case_id"] == case_id].sort_values("slice_index")
    if case_df.empty:
        raise ValueError(f"Case {case_id!r} was not found in {CSV_PATH}.")

    slices: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    has_all_masks = True
    for _, row in case_df.iterrows():
        image_path = IMAGE_DIR / row["filename"]
        if not image_path.exists():
            continue
        with Image.open(image_path) as image:
            raw = np.array(image, dtype=np.float32)
            mode = image.mode
        raw = standardize_dims(raw, 512, 512)
        if raw.max() > 255.0 or mode in ("I;16", "I", "F"):
            normalized = raw / 65535.0
        else:
            normalized = (raw - raw.min()) / max(1e-4, raw.max() - raw.min())
        slices.append(normalized)

        mask_path = MASK_DIR / row["filename"]
        if mask_path.exists():
            with Image.open(mask_path) as mask_image:
                masks.append(standardize_dims(np.array(mask_image, dtype=np.uint8), 512, 512))
        else:
            has_all_masks = False

    if not slices:
        raise ValueError(f"No readable images found for {case_id!r}.")
    ground_truth = np.stack(masks) if has_all_masks and len(masks) == len(slices) else None
    return np.stack(slices), ground_truth, time.perf_counter() - start


def generate_synthetic_case(num_slices: int = 16) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    """Generates synthetic normalized CT slices for benchmarking when dataset is unavailable."""
    start = time.perf_counter()
    np.random.seed(42)
    slices = np.random.uniform(0.1, 0.9, (num_slices, 512, 512)).astype(np.float32)
    return slices, None, time.perf_counter() - start


def compute_metrics(predictions: np.ndarray, targets: Optional[np.ndarray]) -> Optional[Dict[str, Dict[str, float]]]:
    if targets is None:
        return None
    metrics: Dict[str, Dict[str, float]] = {}
    for class_id, class_name in ((1, "kidney"), (2, "tumor"), (3, "cyst")):
        pred = (predictions == class_id)
        target = (targets == class_id)
        tp = int(np.sum(pred & target))
        fp = int(np.sum(pred & ~target))
        fn = int(np.sum(~pred & target))
        dice_denom = 2 * tp + fp + fn
        iou_denom = tp + fp + fn
        metrics[class_name] = {
            "dice": (2.0 * tp / dice_denom) if dice_denom else 1.0,
            "iou": (tp / iou_denom) if iou_denom else 1.0,
            "precision": (tp / (tp + fp)) if (tp + fp) else 0.0,
            "recall": (tp / (tp + fn)) if (tp + fn) else 0.0,
        }
    return metrics


def save_masks(predictions: np.ndarray, directory: Path) -> float:
    start = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    for index, mask in enumerate(predictions):
        Image.fromarray(mask).save(directory / f"mask_{index:04d}.png")
    return time.perf_counter() - start


def benchmark_inference_run(
    name: str,
    backend_type: str,
    model: Any,
    volume: np.ndarray,
    targets: Optional[np.ndarray],
    device: torch.device,
    batch_size: int,
    use_tta: bool,
    use_amp: bool,
    threads: int,
    save_directory: Path,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Executes a single benchmark run and records comprehensive timing and memory metrics."""
    # Configure threads for this run
    if device.type == "cpu" or backend_type != "pytorch_cuda":
        set_cpu_threads(threads)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Warm-up pass is deliberately excluded from reported timings
    warmup_batch = volume[:min(batch_size, len(volume))]
    _ = predict_volume_batched(
        model, warmup_batch, device, use_tta=use_tta,
        batch_size=batch_size, use_amp=use_amp, backend=backend_type
    )
    synchronize(device)

    rss_before = get_current_rss_mb()
    utilization_before = gpu_utilization() if device.type == "cuda" else None

    # Track detailed per-batch inference times to compute average and median
    batch_latencies: List[float] = []
    total_start = time.perf_counter()
    model_start = time.perf_counter()

    if backend_type == "onnx_cpu":
        session = get_onnx_session(num_threads=threads)
        predictions_list = []
        for start_idx in range(0, len(volume), batch_size):
            b_np = np.ascontiguousarray(volume[start_idx:start_idx + batch_size, np.newaxis, :, :], dtype=np.float32)
            t_b_start = time.perf_counter()
            b_pred = predict_tensor(model, b_np, device, use_tta=use_tta, return_numpy=True, backend="onnx_cpu")
            batch_latencies.append(time.perf_counter() - t_b_start)
            predictions_list.append(b_pred)
        raw_predictions = np.concatenate(predictions_list, axis=0) if predictions_list else np.empty((0, *volume.shape[1:]), dtype=np.uint8)
    else:
        # PyTorch path (CUDA or CPU)
        prediction_batches = []
        cuda_events = []
        for start_idx in range(0, len(volume), batch_size):
            b_np = np.ascontiguousarray(volume[start_idx:start_idx + batch_size])
            b_tensor = torch.from_numpy(b_np).unsqueeze(1)
            if device.type == "cuda":
                b_tensor = b_tensor.to(device, non_blocking=True)
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()
            else:
                b_tensor = b_tensor.to(device)
                t_b_start = time.perf_counter()

            pred_b = predict_tensor(
                model, b_tensor, device, use_tta=use_tta,
                return_numpy=False, use_amp=use_amp, backend=backend_type
            )
            prediction_batches.append(pred_b)

            if device.type == "cuda":
                end_ev.record()
                cuda_events.append((start_ev, end_ev))
            else:
                batch_latencies.append(time.perf_counter() - t_b_start)

        if cuda_events:
            synchronize(device)
            batch_latencies = [s.elapsed_time(e) / 1000.0 for s, e in cuda_events]
        raw_predictions = torch.cat(prediction_batches, dim=0).cpu().numpy()

    synchronize(device)
    inference_seconds = time.perf_counter() - model_start
    rss_after = get_current_rss_mb()
    utilization_after = gpu_utilization() if device.type == "cuda" else None

    # Postprocessing
    post_start = time.perf_counter()
    filtered_predictions, filter_stats = filter_prediction_slices(raw_predictions, log_result=False)
    cleaned_predictions = clean_segmentation_volume(filtered_predictions)
    postprocessing_seconds = time.perf_counter() - post_start

    # Mask saving
    mask_saving_seconds = save_masks(cleaned_predictions, save_directory)
    total_seconds = time.perf_counter() - total_start

    total_slices = len(volume)
    avg_slice_time = (sum(batch_latencies) / total_slices) if total_slices else 0.0
    median_batch_time = float(np.median(batch_latencies)) if batch_latencies else 0.0
    median_slice_time = median_batch_time / max(1, batch_size)

    peak_gpu_vram = (torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)) if device.type == "cuda" else None

    result = {
        "name": name,
        "backend": backend_type,
        "device": str(device),
        "batch_size": batch_size,
        "threads": threads,
        "precision": "amp_fp16" if use_amp and device.type == "cuda" else "fp32",
        "tta": use_tta,
        "slices": total_slices,
        "total_end_to_end_seconds": total_seconds,
        "model_and_transfer_seconds": inference_seconds,
        "model_only_seconds": sum(batch_latencies),
        "average_inference_time_per_slice_ms": avg_slice_time * 1000.0,
        "median_inference_time_per_slice_ms": median_slice_time * 1000.0,
        "median_batch_time_ms": median_batch_time * 1000.0,
        "postprocessing_seconds": postprocessing_seconds,
        "mask_saving_seconds": mask_saving_seconds,
        "slices_per_second": (total_slices / total_seconds) if total_seconds > 0 else 0.0,
        "peak_gpu_vram_mib": peak_gpu_vram,
        "peak_cpu_rss_mib": max(rss_before, rss_after),
        "gpu_utilization_before_pct": utilization_before,
        "gpu_utilization_after_pct": utilization_after,
        "filter_stats": filter_stats,
        "metrics": compute_metrics(cleaned_predictions, targets),
    }
    return result, cleaned_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NephroAI Hardware Detection & Model Inference Benchmark (GPU & CPU)."
    )
    parser.add_argument("--case-id", default="case_00110", help="Dataset case ID (default: case_00110).")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic volume if dataset archive is absent.")
    parser.add_argument("--num-slices", type=int, default=16, help="Number of slices for synthetic volume (default: 16).")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4], help="Batch sizes to benchmark (default: 1 4).")
    parser.add_argument("--threads", type=int, nargs="+", default=[2, 4, 8], help="CPU thread counts to evaluate (default: 2 4 8).")
    parser.add_argument("--backend", choices=["auto", "all", "pytorch_cuda", "onnx_cpu", "pytorch_cpu"], default="all",
                        help="Backend to benchmark ('all', 'auto', 'pytorch_cuda', 'onnx_cpu', 'pytorch_cpu').")
    parser.add_argument("--tta", action="store_true", help="Benchmark with 4-Fold Test-Time Augmentation (TTA).")
    parser.add_argument("--compare-fp32", action="store_true", help="Compare FP32 vs AMP on CUDA.")
    parser.add_argument("--output", type=Path, default=Path("reports/inference_benchmark.json"), help="Output JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 80)
    print("NEPHROAI HARDWARE DETECTION & INFERENCE BENCHMARK")
    print("=" * 80)

    # 1. Hardware Detection & Device Selection
    device = get_device()
    auto_backend = get_available_backend()
    default_threads = get_cpu_threads()

    print(f"Detected Execution Device : {device}")
    print(f"Active Default Backend    : {auto_backend}")
    print(f"Default CPU Threads       : {default_threads} (total logical cores: {os.cpu_count()})")
    print("-" * 80)

    # 2. Measure Model Loading Times
    print("Measuring model loading and session initialization times...")
    t0 = time.perf_counter()
    model, _ = load_model()
    pytorch_load_time_sec = time.perf_counter() - t0
    print(f"  PyTorch Model (.pth) Loaded in : {pytorch_load_time_sec:.3f} s")

    onnx_load_time_sec = None
    try:
        t0 = time.perf_counter()
        onnx_session = get_onnx_session()
        onnx_load_time_sec = time.perf_counter() - t0
        print(f"  ONNX Runtime Session Loaded in : {onnx_load_time_sec:.3f} s")
    except Exception as e:
        print(f"  ONNX Runtime Session Unavailable: {e}")

    # 3. Load or Generate CT Volume
    if not args.synthetic and CSV_PATH.exists():
        try:
            print(f"\nLoading patient case: {args.case_id}...")
            volume, targets, preprocessing_seconds = load_case(args.case_id)
            print(f"  Loaded {len(volume)} CT slices ({volume.shape[1]}x{volume.shape[2]}). Preprocessing: {preprocessing_seconds:.3f} s.")
        except Exception as e:
            print(f"  Failed to load case {args.case_id} ({e}). Falling back to synthetic volume.")
            volume, targets, preprocessing_seconds = generate_synthetic_case(args.num_slices)
    else:
        print(f"\nGenerating synthetic CT volume ({args.num_slices} slices, 512x512)...")
        volume, targets, preprocessing_seconds = generate_synthetic_case(args.num_slices)

    # 4. Determine Variants to Benchmark
    variants_to_run = []
    if device.type == "cuda" and (args.backend in ("all", "auto", "pytorch_cuda")):
        for bs in args.batch_sizes:
            for use_amp in ([True, False] if args.compare_fp32 else [True]):
                variants_to_run.append({
                    "name": f"PyTorch CUDA (B={bs}, {'AMP' if use_amp else 'FP32'})",
                    "backend": "pytorch_cuda",
                    "batch_size": bs,
                    "use_amp": use_amp,
                    "threads": default_threads,
                })

    # CPU backends
    if args.backend in ("all", "auto", "onnx_cpu", "pytorch_cpu") or device.type == "cpu":
        threads_to_test = [t for t in args.threads if t <= (os.cpu_count() or 8) * 2]
        if not threads_to_test:
            threads_to_test = [default_threads]

        # ONNX Runtime CPU
        if args.backend in ("all", "auto", "onnx_cpu"):
            for thr in threads_to_test:
                for bs in args.batch_sizes:
                    variants_to_run.append({
                        "name": f"ONNX Runtime CPU (B={bs}, threads={thr})",
                        "backend": "onnx_cpu",
                        "batch_size": bs,
                        "use_amp": False,
                        "threads": thr,
                    })

        # PyTorch CPU Fallback
        if args.backend in ("all", "pytorch_cpu") or (args.backend == "auto" and device.type == "cpu"):
            for thr in threads_to_test:
                for bs in args.batch_sizes:
                    variants_to_run.append({
                        "name": f"PyTorch CPU (B={bs}, threads={thr})",
                        "backend": "pytorch_cpu",
                        "batch_size": bs,
                        "use_amp": True,
                        "threads": thr,
                    })

    # 5. Execute Benchmarks
    results: List[Dict[str, Any]] = []
    baseline_predictions: Optional[np.ndarray] = None

    with tempfile.TemporaryDirectory(prefix="nephroai_bench_") as temp_dir:
        temp_path = Path(temp_dir)
        print(f"\nRunning {len(variants_to_run)} benchmark configuration(s)...")

        for idx, var in enumerate(variants_to_run, 1):
            print(f"[{idx}/{len(variants_to_run)}] Executing: {var['name']}...")
            try:
                res, preds = benchmark_inference_run(
                    name=var["name"],
                    backend_type=var["backend"],
                    model=model,
                    volume=volume,
                    targets=targets,
                    device=device if var["backend"] == "pytorch_cuda" else torch.device("cpu"),
                    batch_size=var["batch_size"],
                    use_tta=args.tta,
                    use_amp=var["use_amp"],
                    threads=var["threads"],
                    save_directory=temp_path / f"var_{idx}",
                )

                if baseline_predictions is None:
                    baseline_predictions = preds
                    res["difference_pixels_vs_baseline"] = 0
                else:
                    diff_voxels = int(np.sum(preds != baseline_predictions))
                    res["difference_pixels_vs_baseline"] = diff_voxels
                    res["prediction_match_pct"] = round(100.0 * (1.0 - (diff_voxels / volume.size)), 5)

                results.append(res)
            except Exception as exc:
                print(f"  [ERROR] Benchmark failed for {var['name']}: {exc}")
                results.append({
                    "name": var["name"],
                    "backend": var["backend"],
                    "status": "error",
                    "error_message": str(exc),
                })

    # 6. Format and Print Comparison Table
    print("\n" + "=" * 110)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 110)
    header = f"{'Configuration':<38} {'Batch':<6} {'Threads':<8} {'Total(s)':<9} {'Model(s)':<9} {'Avg/sl(ms)':<11} {'Med/sl(ms)':<11} {'Slices/s':<9} {'RAM/VRAM':<10}"
    print(header)
    print("-" * 110)

    for r in results:
        if r.get("status") == "error":
            print(f"{r['name']:<38} ERROR: {r.get('error_message')}")
            continue

        mem_str = f"{r['peak_gpu_vram_mib']:.0f}M VRAM" if r["peak_gpu_vram_mib"] else f"{r['peak_cpu_rss_mib']:.0f}M RSS"
        print(
            f"{r['name']:<38} {r['batch_size']:<6} {r['threads']:<8} "
            f"{r['total_end_to_end_seconds']:<9.2f} {r['model_only_seconds']:<9.2f} "
            f"{r['average_inference_time_per_slice_ms']:<11.1f} {r['median_inference_time_per_slice_ms']:<11.1f} "
            f"{r['slices_per_second']:<9.2f} {mem_str:<10}"
        )
    print("=" * 110)

    # 7. Speedup Calculations
    pytorch_cpu_runs = [r for r in results if r.get("backend") == "pytorch_cpu" and "model_only_seconds" in r]
    onnx_cpu_runs = [r for r in results if r.get("backend") == "onnx_cpu" and "model_only_seconds" in r]
    cuda_runs = [r for r in results if r.get("backend") == "pytorch_cuda" and "model_only_seconds" in r]

    if pytorch_cpu_runs and onnx_cpu_runs:
        print("\n--- CPU Backend Speedup Comparison (PyTorch CPU vs ONNX Runtime CPU) ---")
        for ort_r in onnx_cpu_runs:
            matching_pt = [
                pt for pt in pytorch_cpu_runs
                if pt["batch_size"] == ort_r["batch_size"] and pt["threads"] == ort_r["threads"]
            ]
            if matching_pt:
                pt_time = matching_pt[0]["model_only_seconds"]
                ort_time = ort_r["model_only_seconds"]
                speedup = pt_time / max(1e-4, ort_time)
                pct_reduction = (pt_time - ort_time) / pt_time * 100.0
                print(
                    f"Threads={ort_r['threads']}, Batch={ort_r['batch_size']}: "
                    f"PyTorch CPU={pt_time:.2f}s ({matching_pt[0]['average_inference_time_per_slice_ms']:.1f}ms/sl) -> "
                    f"ONNX CPU={ort_time:.2f}s ({ort_r['average_inference_time_per_slice_ms']:.1f}ms/sl) | "
                    f"Speedup: {speedup:.2f}x ({pct_reduction:.1f}% latency reduction)"
                )

    if cuda_runs and pytorch_cpu_runs:
        print("\n--- GPU vs CPU Acceleration ---")
        fastest_gpu = min(cuda_runs, key=lambda x: x["model_only_seconds"])
        fastest_cpu = min(onnx_cpu_runs or pytorch_cpu_runs, key=lambda x: x["model_only_seconds"])
        gpu_speedup = fastest_cpu["model_only_seconds"] / max(1e-4, fastest_gpu["model_only_seconds"])
        print(f"Fastest GPU ({fastest_gpu['name']}): {fastest_gpu['model_only_seconds']:.2f}s")
        print(f"Fastest CPU ({fastest_cpu['name']}): {fastest_cpu['model_only_seconds']:.2f}s")
        print(f"GPU Hardware Acceleration: {gpu_speedup:.2f}x faster than CPU")

    # 8. Save Full JSON Report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": str(device),
            "cpu_count": os.cpu_count(),
            "default_cpu_threads": default_threads,
            "gpu_detected": (device.type == "cuda"),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "model_loading_seconds": {
            "pytorch_pth": pytorch_load_time_sec,
            "onnx_session": onnx_load_time_sec,
        },
        "case_id": args.case_id if not args.synthetic else "synthetic",
        "slices_processed": len(volume),
        "preprocessing_seconds": preprocessing_seconds,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nDetailed JSON report saved to: {args.output.resolve()}\n")


if __name__ == "__main__":
    main()
