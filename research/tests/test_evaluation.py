"""Checks using artificial inputs only; no study records are included."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dolphin_behavior_mil.evaluation import (  # noqa: E402
    adaptive_ece,
    campaign_cluster_bootstrap,
    holm_adjust,
    macro_average_precision,
    paired_campaign_prediction_swap_test,
    percentile_interval,
    select_largest_max_f1_threshold,
)


class EvaluationMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.truth = np.asarray(
            [
                [1, 0],
                [1, 1],
                [0, 1],
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
                [0, 0],
            ],
            dtype=np.int8,
        )
        self.good = np.asarray(
            [
                [0.9, 0.1],
                [0.8, 0.8],
                [0.2, 0.9],
                [0.1, 0.2],
                [0.7, 0.3],
                [0.3, 0.7],
                [0.8, 0.8],
                [0.2, 0.1],
            ]
        )
        self.bad = 1.0 - self.good
        self.campaigns = ["a", "a", "b", "b", "c", "c", "d", "d"]

    def test_macro_average_precision_orders_predictions(self) -> None:
        self.assertGreater(
            macro_average_precision(self.truth, self.good),
            macro_average_precision(self.truth, self.bad),
        )

    def test_f1_threshold_uses_largest_tie(self) -> None:
        result = select_largest_max_f1_threshold(
            np.asarray([1, 1, 0, 0]),
            np.asarray([0.8, 0.8, 0.2, 0.1]),
        )
        self.assertEqual(result, {"threshold": 0.8, "f1": 1.0})

    def test_adaptive_ece_is_zero_for_exact_group_calibration(self) -> None:
        self.assertAlmostEqual(
            adaptive_ece(
                np.asarray([0, 0, 1, 1]),
                np.asarray([0.0, 0.0, 1.0, 1.0]),
                bin_count=2,
            ),
            0.0,
        )

    def test_cluster_bootstrap_is_deterministic_and_finite(self) -> None:
        first = campaign_cluster_bootstrap(
            self.truth,
            self.good,
            self.campaigns,
            resamples=50,
            seed=11,
        )
        second = campaign_cluster_bootstrap(
            self.truth,
            self.good,
            self.campaigns,
            resamples=50,
            seed=11,
        )
        np.testing.assert_array_equal(first, second)
        lower, upper = percentile_interval(first)
        self.assertLessEqual(lower, upper)
        self.assertTrue(0 <= lower <= 1 and 0 <= upper <= 1)

    def test_paired_cluster_swap_detects_direction_and_is_deterministic(self) -> None:
        first = paired_campaign_prediction_swap_test(
            self.truth,
            self.good,
            self.bad,
            self.campaigns,
            permutations=100,
            seed=19,
        )
        second = paired_campaign_prediction_swap_test(
            self.truth,
            self.good,
            self.bad,
            self.campaigns,
            permutations=100,
            seed=19,
        )
        self.assertEqual(first, second)
        self.assertGreater(first.observed_delta, 0)
        self.assertGreaterEqual(first.p_value, 1 / 101)

    def test_holm_adjustment_is_monotone_in_sorted_order(self) -> None:
        adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.9})
        self.assertEqual(adjusted["a"], 0.04)
        self.assertEqual(adjusted["c"], 0.09)
        self.assertEqual(adjusted["b"], 0.09)
        self.assertEqual(adjusted["d"], 0.9)


if __name__ == "__main__":
    unittest.main()
