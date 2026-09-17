"""Tests for clustered uncertainty and sample grouping in Research Review."""

import sys
import unittest
from pathlib import Path

import numpy as np


REVIEW_ROOT = Path(__file__).resolve().parents[1] / "research_review"
sys.path.insert(0, str(REVIEW_ROOT))

from run_review import cluster_bootstrap_mean, prediction_shape, video_groups  # noqa: E402


class ResearchReviewTests(unittest.TestCase):
    def test_video_key_uses_first_id_component(self):
        ids = np.asarray(["v1|0|1", "v1|1|2", "v2|0|1"])
        self.assertEqual(video_groups(ids).tolist(), ["v1", "v1", "v2"])

    def test_cluster_bootstrap_is_deterministic(self):
        values = np.asarray([1.0, 3.0, 2.0, 6.0])
        groups = np.asarray(["a", "a", "b", "c"])
        a = cluster_bootstrap_mean(values, groups, 100, 42)
        b = cluster_bootstrap_mean(values, groups, 100, 42)
        self.assertEqual(a, b)
        self.assertEqual(a["n_videos"], 3)

    def test_prediction_shape_detects_contraction(self):
        full = np.asarray([-2.0, -1.0, 1.0, 2.0])
        missing = 0.5 * full + 0.25
        result = prediction_shape(full, missing)
        self.assertAlmostEqual(result["prediction_std_ratio"], 0.5)
        self.assertAlmostEqual(result["full_to_missing_slope"], 0.5)
        self.assertAlmostEqual(result["prediction_mean_shift"], 0.25)


if __name__ == "__main__":
    unittest.main()
