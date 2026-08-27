import unittest

import numpy as np

from src.mlem import chi2_reduced, normalize_sum, run_mlem, smooth_121
from src.response_matrix import parse_etrue_header


class CoreMathTests(unittest.TestCase):
    def test_parse_etrue_header(self):
        self.assertEqual(parse_etrue_header("Etrue_20p000000_keV"), 20.0)
        self.assertEqual(parse_etrue_header("Etrue_6000p500000_keV"), 6000.5)

    def test_smooth_121_preserves_constant_vector(self):
        values = np.ones(5)
        np.testing.assert_allclose(smooth_121(values), values)

    def test_chi2_reduced_zero_for_perfect_prediction(self):
        y = np.array([1.0, 4.0, 9.0])
        self.assertEqual(chi2_reduced(y, y), 0.0)

    def test_mlem_identity_recovers_measurement(self):
        h = np.eye(3)
        y = np.array([2.0, 5.0, 7.0])
        result = run_mlem(y, h, n_iter=5)
        np.testing.assert_allclose(result.x, y, rtol=1e-10, atol=1e-10)

    def test_normalize_sum(self):
        np.testing.assert_allclose(normalize_sum(np.array([2.0, 2.0])), [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()

