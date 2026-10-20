import unittest

import numpy as np

from backend.inference_engine import filter_predictions


class KidneyAbsenceFilterTests(unittest.TestCase):
    def test_background_only_mask_is_unchanged(self):
        mask = np.zeros((3, 4), dtype=np.uint8)

        filtered = filter_predictions(mask)

        np.testing.assert_array_equal(filtered, mask)

    def test_empty_mask_is_unchanged(self):
        mask = np.zeros((0, 0), dtype=np.uint8)

        filtered = filter_predictions(mask)

        self.assertEqual(filtered.shape, mask.shape)
        np.testing.assert_array_equal(filtered, mask)

    def test_kidney_present_mask_is_unchanged(self):
        mask = np.array([[0, 1, 2], [3, 1, 0]], dtype=np.uint8)

        filtered = filter_predictions(mask)

        np.testing.assert_array_equal(filtered, mask)

    def test_no_kidney_tumor_pixels_become_background(self):
        mask = np.array([[0, 2, 0], [2, 0, 0]], dtype=np.uint8)

        filtered = filter_predictions(mask)

        np.testing.assert_array_equal(filtered, np.zeros_like(mask))

    def test_no_kidney_cyst_pixels_become_background(self):
        mask = np.array([[0, 3, 0], [3, 0, 0]], dtype=np.uint8)

        filtered = filter_predictions(mask)

        np.testing.assert_array_equal(filtered, np.zeros_like(mask))

    def test_no_kidney_tumor_and_cyst_pixels_become_background(self):
        mask = np.array([[2, 3, 0], [3, 2, 0]], dtype=np.uint8)

        filtered = filter_predictions(mask)

        np.testing.assert_array_equal(filtered, np.zeros_like(mask))

    def test_original_mask_is_not_modified(self):
        mask = np.array([[0, 2], [3, 0]], dtype=np.uint8)
        original = mask.copy()

        filtered = filter_predictions(mask)

        np.testing.assert_array_equal(mask, original)
        self.assertFalse(np.shares_memory(mask, filtered))


if __name__ == "__main__":
    unittest.main()
