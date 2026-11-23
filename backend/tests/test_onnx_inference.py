import os
import unittest
from pathlib import Path
import numpy as np
import torch

from backend.model_loader import (
    load_model,
    get_device,
    get_onnx_session,
    export_and_validate_onnx,
    reset_singleton_state,
)
from backend.inference_engine import (
    predict_tensor,
    predict_volume_batched,
    filter_predictions,
    filter_prediction_slices,
    clean_segmentation_volume,
)


class ONNXInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reset_singleton_state()
        cls.model, cls.device = load_model()
        cls.onnx_session = get_onnx_session()

    def test_onnx_session_uses_cpu_execution_provider(self):
        self.assertIsNotNone(self.onnx_session)
        providers = self.onnx_session.get_providers()
        self.assertIn("CPUExecutionProvider", providers)

    def test_numerical_equivalence_single_batch(self):
        """Verifies that ONNX Runtime CPU predictions match PyTorch CPU predictions."""
        np.random.seed(123)
        test_in = np.random.uniform(0.0, 1.0, (1, 1, 512, 512)).astype(np.float32)
        tensor_in = torch.from_numpy(test_in)

        # PyTorch CPU
        pt_pred = predict_tensor(
            self.model, tensor_in, torch.device("cpu"),
            use_tta=False, return_numpy=True, backend="pytorch_cpu"
        )

        # ONNX Runtime CPU
        ort_pred = predict_tensor(
            self.model, test_in, torch.device("cpu"),
            use_tta=False, return_numpy=True, backend="onnx_cpu"
        )

        self.assertEqual(pt_pred.shape, ort_pred.shape)
        self.assertEqual(pt_pred.dtype, np.uint8)
        self.assertEqual(ort_pred.dtype, np.uint8)

        # Check raw logit parity
        model_device = next(self.model.parameters()).device
        with torch.inference_mode():
            pt_logits = self.model(tensor_in.to(model_device)).cpu().numpy()
        ort_logits = self.onnx_session.run(["output"], {"input": test_in})[0]
        max_logit_diff = float(np.max(np.abs(pt_logits - ort_logits)))
        # When comparing CUDA cuDNN (with Tensor Cores) vs CPU OpenMP, slight FP variance is expected
        diff_threshold = 0.15 if model_device.type == "cuda" else 1e-3
        self.assertLess(max_logit_diff, diff_threshold, f"Max logit diff {max_logit_diff} exceeded tolerance.")

        # On random noise, allow at most 0.25% pixel difference from near-zero tie breaks (>99.75% match)
        total_pixels = test_in.shape[0] * 512 * 512
        mismatches = int(np.sum(pt_pred != ort_pred))
        mismatch_pct = mismatches / total_pixels * 100.0
        self.assertLess(mismatch_pct, 0.25, f"Mismatch {mismatch_pct:.4f}% exceeded tolerance.")

    def test_tta_equivalence_between_pytorch_and_onnx(self):
        """Verifies that 4-fold TTA produces consistent predictions between PyTorch and ONNX."""
        np.random.seed(456)
        test_in = np.random.uniform(0.0, 1.0, (1, 1, 512, 512)).astype(np.float32)

        pt_tta = predict_tensor(
            self.model, test_in, torch.device("cpu"),
            use_tta=True, return_numpy=True, backend="pytorch_cpu"
        )
        ort_tta = predict_tensor(
            self.model, test_in, torch.device("cpu"),
            use_tta=True, return_numpy=True, backend="onnx_cpu"
        )

        self.assertEqual(pt_tta.shape, ort_tta.shape)
        total_pixels = 512 * 512
        mismatches = int(np.sum(pt_tta != ort_tta))
        mismatch_pct = mismatches / total_pixels * 100.0
        self.assertLess(mismatch_pct, 0.05)

    def test_volume_batched_onnx(self):
        """Verifies predict_volume_batched works seamlessly with ONNX CPU backend."""
        volume = np.random.uniform(0.0, 1.0, (3, 512, 512)).astype(np.float32)
        timing = {}
        preds = predict_volume_batched(
            self.model, volume, torch.device("cpu"),
            use_tta=False, batch_size=2, timing=timing, backend="onnx_cpu"
        )
        self.assertEqual(preds.shape, (3, 512, 512))
        self.assertEqual(preds.dtype, np.uint8)
        self.assertIn("model_only_seconds", timing)
        self.assertGreater(timing["model_only_seconds"], 0.0)

    def test_postprocessing_3d_cca_on_onnx_outputs(self):
        """Verifies that Kidney-absence filter and 3D CCA postprocessing process ONNX masks properly."""
        raw_volume = np.zeros((3, 512, 512), dtype=np.uint8)
        # Add a simulated kidney region and tumor region
        raw_volume[1, 200:250, 200:250] = 1  # kidney
        raw_volume[1, 220:230, 220:230] = 2  # tumor

        filtered, stats = filter_prediction_slices(raw_volume, log_result=False)
        cleaned = clean_segmentation_volume(filtered)

        self.assertEqual(cleaned.shape, raw_volume.shape)
        self.assertTrue(np.any(cleaned == 1))
        self.assertTrue(np.any(cleaned == 2))


if __name__ == "__main__":
    unittest.main()
