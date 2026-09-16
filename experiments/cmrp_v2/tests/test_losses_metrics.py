import sys
import unittest
from pathlib import Path

import numpy as np
import torch


V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

from cmrp_v2.losses import (  # noqa: E402
    pointwise_consistency_loss,
    relational_consistency_loss,
)
from cmrp_v2.metrics import (  # noqa: E402
    cosine_drift,
    median_pair_distance,
    relational_drift_rms,
    representation_drift_metrics,
)


class LossTests(unittest.TestCase):
    def test_identical_views_have_zero_loss(self):
        z = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        self.assertAlmostEqual(float(pointwise_consistency_loss(z, z)), 0.0, places=7)
        self.assertAlmostEqual(float(relational_consistency_loss(z, z)), 0.0, places=7)

    def test_relational_loss_is_rotation_invariant(self):
        z = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        rotated = z @ rotation
        self.assertAlmostEqual(float(relational_consistency_loss(z, rotated)), 0.0, places=6)
        self.assertGreater(float(pointwise_consistency_loss(z, rotated)), 0.1)

    def test_stop_gradient_only_on_full_view(self):
        full = torch.randn(4, 3, requires_grad=True)
        missing = torch.randn(4, 3, requires_grad=True)
        relational_consistency_loss(full, missing).backward()
        self.assertIsNone(full.grad)
        self.assertIsNotNone(missing.grad)

    def test_single_sample_relational_loss_is_differentiable_zero(self):
        full = torch.randn(1, 3, requires_grad=True)
        missing = torch.randn(1, 3, requires_grad=True)
        loss = relational_consistency_loss(full, missing)
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        self.assertIsNotNone(missing.grad)


class MetricTests(unittest.TestCase):
    def test_identical_views_have_zero_drift(self):
        z = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        self.assertAlmostEqual(cosine_drift(z, z), 0.0, places=12)
        self.assertAlmostEqual(relational_drift_rms(z, z, block_size=2), 0.0, places=12)

    def test_exact_median_pair_distance(self):
        z = np.array([[0.0], [1.0], [3.0]])
        self.assertAlmostEqual(median_pair_distance(z, block_size=2), 2.0, places=12)

    def test_relational_metric_is_rotation_invariant(self):
        z = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
        self.assertAlmostEqual(
            relational_drift_rms(z, z @ rotation, block_size=2), 0.0, places=12
        )

    def test_shared_scale_is_used_verbatim(self):
        full = np.array([[1.0, 0.0], [0.0, 1.0]])
        missing = np.zeros_like(full)
        metrics = representation_drift_metrics(
            full, missing, shared_reference_scale=2.0, block_size=2
        )
        self.assertAlmostEqual(metrics["absolute_l2"], 1.0, places=12)
        self.assertAlmostEqual(metrics["shared_reference_D_cross"], 0.5, places=12)


if __name__ == "__main__":
    unittest.main()
