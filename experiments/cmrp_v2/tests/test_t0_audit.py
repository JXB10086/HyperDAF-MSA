"""Focused tests for T0 numerical and split-discipline primitives."""

import sys
import unittest
from pathlib import Path

import numpy as np


T0_ROOT = Path(__file__).resolve().parents[1] / "t0_mechanism_audit"
sys.path.insert(0, str(T0_ROOT))

from audit_lib import (  # noqa: E402
    drift_projection,
    orthogonal_procrustes,
    select_ridge_alpha,
)
from run_t0 import assert_aligned_ids  # noqa: E402


class ProcrustesTests(unittest.TestCase):
    def test_recovers_known_rotation_on_unseen_rows(self):
        rng = np.random.default_rng(7)
        q0, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        full = rng.normal(size=(60, 8))
        missing = full @ q0.T
        q = orthogonal_procrustes(missing[:40], full[:40])
        np.testing.assert_allclose(missing[40:] @ q, full[40:], atol=1e-10)
        np.testing.assert_allclose(q.T @ q, np.eye(8), atol=1e-10)

    def test_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            orthogonal_procrustes(np.zeros((3, 2)), np.zeros((4, 2)))


class GradientProjectionTests(unittest.TestCase):
    def test_zero_norm_is_finite_and_counted(self):
        result = drift_projection(
            np.zeros((2, 3)), np.asarray([[0, 0, 0], [1, 0, 0]]), np.zeros((2, 3))
        )
        self.assertTrue(np.isfinite(result["absolute_cosine"]).all())
        self.assertEqual(result["zero_denominator"].tolist(), [True, True])

    def test_parallel_direction_has_unit_absolute_cosine(self):
        result = drift_projection(
            np.zeros((1, 2)), np.asarray([[2.0, 0.0]]), np.asarray([[-3.0, 0.0]])
        )
        self.assertAlmostEqual(float(result["absolute_cosine"][0]), 1.0)
        self.assertAlmostEqual(float(result["absolute_projection"][0]), 6.0)


class ProbeDisciplineTests(unittest.TestCase):
    def test_validation_selects_without_using_test(self):
        rng = np.random.default_rng(11)
        x_train = rng.normal(size=(40, 5))
        y_train = x_train[:, 0] + rng.normal(scale=0.1, size=40)
        x_valid = rng.normal(size=(20, 5))
        y_valid = x_valid[:, 0] + rng.normal(scale=0.1, size=20)
        model_a, trace_a = select_ridge_alpha(
            x_train, y_train, x_valid, y_valid, [0.0, 1.0, 100.0]
        )
        model_b, trace_b = select_ridge_alpha(
            x_train, y_train, x_valid, y_valid, [0.0, 1.0, 100.0]
        )
        self.assertEqual(model_a.alpha, model_b.alpha)
        self.assertEqual(trace_a, trace_b)

    def test_validation_perturbation_can_change_selection(self):
        rng = np.random.default_rng(19)
        x_train = rng.normal(size=(12, 20))
        y_train = rng.normal(size=12)
        x_valid = rng.normal(size=(20, 20))
        clean = x_valid[:, 0]
        noisy = rng.normal(size=20)
        _, trace_clean = select_ridge_alpha(
            x_train, y_train, x_valid, clean, [0.0, 1000.0]
        )
        _, trace_noisy = select_ridge_alpha(
            x_train, y_train, x_valid, noisy, [0.0, 1000.0]
        )
        self.assertNotEqual(trace_clean, trace_noisy)


class AlignmentTests(unittest.TestCase):
    def test_identical_order_passes(self):
        ids = np.asarray(["a", "b", "c"])
        assert_aligned_ids(ids, ids.copy(), "synthetic")

    def test_permutation_fails(self):
        with self.assertRaises(RuntimeError):
            assert_aligned_ids(
                np.asarray(["a", "b"]), np.asarray(["b", "a"]), "synthetic"
            )


if __name__ == "__main__":
    unittest.main()
