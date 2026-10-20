"""Reproducible GPU inference benchmark for the 2D kidney CT pipeline.

Example:
    python -m backend.benchmark_inference --case-id case_00110 --tta

The benchmark compares the pre-optimization transfer pattern (one CPU transfer
per batch) against the deployed deferred-transfer implementation. It does not
alter model weights, predictions, or uploaded-case data.
"""

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
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
from backend.model_loader import load_model


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def gpu_utilization() -> Optional[float]:
    """Read instantaneous utilization when nvidia-smi is available."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return float(output.strip().splitlines()[0])
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def load_case(case_id: str) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    """Load the same 512 x 512 normalized slices used by production inference."""
    start = time.perf_counter()
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


def compute_metrics(predictions: np.ndarray, targets: Optional[np.ndarray]) -> Optional[Dict[str, Dict[str, float]]]:
    if targets is None:
        return None
    metrics: Dict[str, Dict[str, float]] = {}
    for class_id, class_name in ((1, "kidney"), (2, "tumor"), (3, "cyst")):
        pred = predictions == class_id
        target = targets == class_id
        true_positive = int(np.sum(pred & target))
        false_positive = int(np.sum(pred & ~target))
        false_negative = int(np.sum(~pred & target))
        dice_denominator = 2 * true_positive + false_positive + false_negative
        iou_denominator = true_positive + false_positive + false_negative
        metrics[class_name] = {
            "dice": (2 * true_positive / dice_denominator) if dice_denominator else 1.0,
            "iou": (true_positive / iou_denominator) if iou_denominator else 1.0,
            "precision": (true_positive / (true_positive + false_positive)) if (true_positive + false_positive) else 0.0,
            "recall": (true_positive / (true_positive + false_negative)) if (true_positive + false_negative) else 0.0,
        }
    return metrics


def save_masks(predictions: np.ndarray, directory: Path) -> float:
    start = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    for index, mask in enumerate(predictions):
        Image.fromarray(mask).save(directory / f"mask_{index:04d}.png")
    return time.perf_counter() - start


def original_batch_transfer(
    model: torch.nn.Module,
    volume: np.ndarray,
    device: torch.device,
    batch_size: int,
    use_tta: bool,
    use_amp: bool,
    timing: Dict[str, float],
) -> np.ndarray:
    """The original production behavior: convert every predicted batch to NumPy."""
    batches: List[np.ndarray] = []
    cuda_events = []
    with torch.inference_mode():
        for start in range(0, len(volume), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(volume[start:start + batch_size])).unsqueeze(1)
            batch = batch.to(device, non_blocking=(device.type == "cuda"))
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                predicted = predict_tensor(model, batch, device, use_tta=use_tta, return_numpy=False, use_amp=use_amp)
                end_event.record()
                cuda_events.append((start_event, end_event))
                batches.append(predicted.cpu().numpy())
            else:
                model_start = time.perf_counter()
                batches.append(predict_tensor(model, batch, device, use_tta=use_tta, use_amp=use_amp))
                timing["model_only_seconds"] = timing.get("model_only_seconds", 0.0) + (time.perf_counter() - model_start)
    if cuda_events:
        synchronize(device)
        timing["model_only_seconds"] = sum(start.elapsed_time(end) for start, end in cuda_events) / 1000.0
    return np.concatenate(batches, axis=0)


def run_variant(
    name: str,
    model: torch.nn.Module,
    volume: np.ndarray,
    targets: Optional[np.ndarray],
    device: torch.device,
    batch_size: int,
    use_tta: bool,
    use_amp: bool,
    defer_transfers: bool,
    save_directory: Path,
) -> Tuple[Dict[str, Any], np.ndarray]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Warm-up is deliberately excluded from all reported timings.
    warmup = volume[:min(batch_size, len(volume))]
    _ = predict_volume_batched(model, warmup, device, use_tta=use_tta, batch_size=batch_size, use_amp=use_amp)
    synchronize(device)

    utilization_before = gpu_utilization()
    total_start = time.perf_counter()
    model_start = time.perf_counter()
    model_timing: Dict[str, float] = {}
    if defer_transfers:
        raw_predictions = predict_volume_batched(
            model, volume, device, use_tta=use_tta, batch_size=batch_size, use_amp=use_amp,
            timing=model_timing
        )
    else:
        raw_predictions = original_batch_transfer(model, volume, device, batch_size, use_tta, use_amp, model_timing)
    synchronize(device)
    inference_seconds = time.perf_counter() - model_start
    if raw_predictions.shape != volume.shape:
        raise RuntimeError(
            f"Inference changed volume dimensions: expected {volume.shape}, got {raw_predictions.shape}."
        )

    post_start = time.perf_counter()
    filtered_predictions, filter_stats = filter_prediction_slices(raw_predictions, log_result=False)
    cleaned_predictions = clean_segmentation_volume(filtered_predictions)
    postprocessing_seconds = time.perf_counter() - post_start
    mask_saving_seconds = save_masks(cleaned_predictions, save_directory)
    total_seconds = time.perf_counter() - total_start
    utilization_after = gpu_utilization()

    result = {
        "name": name,
        "batch_size": batch_size,
        "precision": "amp_fp16" if use_amp and device.type == "cuda" else "fp32",
        "tta": use_tta,
        "slices": int(len(volume)),
        "all_slices_processed": bool(cleaned_predictions.shape[0] == volume.shape[0]),
        "output_mask_dimensions": list(cleaned_predictions.shape[1:]),
        "total_end_to_end_seconds": total_seconds,
        "model_and_transfer_seconds": inference_seconds,
        "model_only_seconds": model_timing.get("model_only_seconds"),
        "postprocessing_seconds": postprocessing_seconds,
        "mask_saving_seconds": mask_saving_seconds,
        "slices_per_second": len(volume) / total_seconds,
        "average_seconds_per_slice": total_seconds / len(volume),
        "peak_gpu_vram_mib": (torch.cuda.max_memory_allocated(device) / 1024 ** 2) if device.type == "cuda" else None,
        "gpu_utilization_before_pct": utilization_before,
        "gpu_utilization_after_pct": utilization_after,
        "filter_stats": filter_stats,
        "metrics": compute_metrics(cleaned_predictions, targets),
    }
    return result, cleaned_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark kidney CT inference without changing production outputs.")
    parser.add_argument("--case-id", required=True, help="Dataset case ID, for example case_00110.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--tta", action="store_true", help="Match production GPU TTA behavior.")
    parser.add_argument("--compare-fp32", action="store_true", help="Also measure FP32 for each batch size.")
    parser.add_argument("--output", type=Path, default=Path("reports/inference_benchmark.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, device = load_model()
    volume, targets, preprocessing_seconds = load_case(args.case_id)
    if device.type != "cuda":
        print("[WARNING] CUDA is unavailable; GPU VRAM/utilization measurements will be empty.")

    results: List[Dict[str, Any]] = []
    baseline_predictions: Optional[np.ndarray] = None
    with tempfile.TemporaryDirectory(prefix="nephroai_benchmark_") as temporary_dir:
        temporary_path = Path(temporary_dir)
        baseline, baseline_predictions = run_variant(
            "original_per_batch_transfer", model, volume, targets, device, 16,
            args.tta, True, False, temporary_path / "baseline"
        )
        results.append(baseline)

        for batch_size in args.batch_sizes:
            for use_amp in ([True, False] if args.compare_fp32 and device.type == "cuda" else [True]):
                try:
                    result, predictions = run_variant(
                        "optimized_deferred_transfer", model, volume, targets, device, batch_size,
                        args.tta, use_amp, True, temporary_path / f"optimized_b{batch_size}_{'amp' if use_amp else 'fp32'}"
                    )
                    result["prediction_difference_pixels_vs_original"] = int(np.sum(predictions != baseline_predictions))
                    result["prediction_dimensions_match"] = bool(predictions.shape == baseline_predictions.shape)
                    results.append(result)
                except torch.cuda.OutOfMemoryError:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    results.append({"name": "optimized_deferred_transfer", "batch_size": batch_size, "status": "cuda_oom"})

    report = {
        "case_id": args.case_id,
        "device": str(device),
        "preprocessing_seconds": preprocessing_seconds,
        "slices_loaded": int(len(volume)),
        "slice_dimensions": list(volume.shape[1:]),
        "ground_truth_available": targets is not None,
        "results": results,
    }
    for result in results:
        if "total_end_to_end_seconds" in result:
            result["total_with_preprocessing_seconds"] = result["total_end_to_end_seconds"] + preprocessing_seconds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nVariant                          Batch  Precision  Total(s)  Model(s)  Post(s)  Save(s)  Slices/s  Peak VRAM(MiB)  Diff px")
    for result in results:
        if result.get("status") == "cuda_oom":
            print(f"{result['name']:<32} {result['batch_size']:>5}  OOM")
            continue
        peak_vram = "-" if result["peak_gpu_vram_mib"] is None else f"{result['peak_gpu_vram_mib']:.1f}"
        print(
            f"{result['name']:<32} {result['batch_size']:>5}  {result['precision']:<9} "
            f"{result['total_end_to_end_seconds']:>8.2f}  {result['model_only_seconds']:>8.2f}  "
            f"{result['postprocessing_seconds']:>7.2f}  {result['mask_saving_seconds']:>7.2f}  "
            f"{result['slices_per_second']:>8.2f}  {peak_vram:>14}  "
            f"{result.get('prediction_difference_pixels_vs_original', 0):>7}"
        )
    print(json.dumps(report, indent=2))
    print(f"\nSaved benchmark report to {args.output.resolve()}")


if __name__ == "__main__":
    main()
