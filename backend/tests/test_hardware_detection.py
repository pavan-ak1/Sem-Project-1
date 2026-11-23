import os
import unittest
from unittest.mock import patch, MagicMock
import torch

from backend.model_loader import (
    get_device,
    get_cpu_threads,
    set_cpu_threads,
    get_available_backend,
    reset_singleton_state,
)


class HardwareDetectionTests(unittest.TestCase):
    def setUp(self):
        reset_singleton_state()

    def tearDown(self):
        reset_singleton_state()
        if "INFERENCE_BACKEND" in os.environ:
            del os.environ["INFERENCE_BACKEND"]
        if "CPU_NUM_THREADS" in os.environ:
            del os.environ["CPU_NUM_THREADS"]
        if "OMP_NUM_THREADS" in os.environ:
            del os.environ["OMP_NUM_THREADS"]

    def test_cpu_detection_when_cuda_unavailable(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            device = get_device()
            self.assertEqual(device.type, "cpu")

    def test_graceful_cpu_fallback_when_cuda_initialization_fails(self):
        # Simulate CUDA available but tensor allocation throws RuntimeError (e.g. driver error / OOM)
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch("torch.zeros", side_effect=RuntimeError("CUDA driver failure")):
            device = get_device()
            self.assertEqual(device.type, "cpu")

    def test_cuda_detection_when_available_and_operational(self):
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch("torch.zeros", return_value=MagicMock()), \
             patch.object(torch.cuda, "get_device_name", return_value="NVIDIA RTX 4090"):
            device = get_device()
            self.assertEqual(device.type, "cuda")

    def test_get_cpu_threads_respects_environment(self):
        os.environ["CPU_NUM_THREADS"] = "6"
        self.assertEqual(get_cpu_threads(), 6)

        del os.environ["CPU_NUM_THREADS"]
        os.environ["OMP_NUM_THREADS"] = "3"
        self.assertEqual(get_cpu_threads(), 3)

    def test_set_cpu_threads_updates_thread_count(self):
        set_cpu_threads(5)
        self.assertEqual(get_cpu_threads(), 5)

    def test_get_available_backend_auto_cpu(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            backend = get_available_backend()
            # If onnxruntime is installed, auto backend on CPU is onnx_cpu
            self.assertIn(backend, ("onnx_cpu", "pytorch_cpu"))

    def test_get_available_backend_override(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            os.environ["INFERENCE_BACKEND"] = "pytorch_cpu"
            self.assertEqual(get_available_backend(), "pytorch_cpu")

            os.environ["INFERENCE_BACKEND"] = "onnx_cpu"
            self.assertEqual(get_available_backend(), "onnx_cpu")


if __name__ == "__main__":
    unittest.main()
