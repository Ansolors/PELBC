"""Checks using artificial inputs only; no study records are included."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dolphin_behavior_mil.statistical_analysis import (  # noqa: E402
    bootstrap_average_precision,
    decision_metrics,
    draw_valid_campaign_bootstrap_plan,
    probability_metrics,
)


class StatisticalAnalysisTests(unittest.TestCase):
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
        self.probability = np.asarray(
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
        self.campaigns = ["a", "a", "b", "b", "c", "c", "d", "d"]

    def test_probability_metrics_have_expected_perfect_discrimination(self) -> None:
        result = probability_metrics(self.truth, self.probability, ("x", "y"))
        self.assertEqual(result["macro_average_precision"], 1.0)
        self.assertEqual(result["macro_roc_auc"], 1.0)
        self.assertGreater(result["macro_brier_score"], 0.0)
        self.assertEqual(result["positive_counts"], {"x": 4, "y": 4})

    def test_probability_metrics_report_undefined_roc_explicitly(self) -> None:
        result = probability_metrics(
            self.truth,
            self.probability,
            ("x", "y"),
            roc_auc_minimum_positive=5,
        )
        self.assertIsNone(result["macro_roc_auc"])
        self.assertFalse(result["per_label_roc_auc"]["x"]["defined"])
        self.assertIsNotNone(
            result["per_label_roc_auc"]["x"]["undefined_reason"]
        )

    def test_probability_metrics_report_absent_positive_ap_explicitly(self) -> None:
        truth = self.truth.copy()
        truth[:, 1] = 0
        result = probability_metrics(truth, self.probability, ("x", "y"))
        self.assertIsNone(result["macro_average_precision"])
        self.assertIsNone(result["per_label_average_precision"]["y"])
        self.assertFalse(
            result["per_label_average_precision_status"]["y"]["defined"]
        )

    def test_decision_metrics_accept_row_specific_thresholds(self) -> None:
        thresholds = np.full_like(self.probability, 0.5)
        result = decision_metrics(
            self.truth, self.probability, thresholds, ("x", "y")
        )
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["hamming_loss"], 0.0)

    def test_campaign_plan_and_ap_replicates_are_deterministic(self) -> None:
        first = draw_valid_campaign_bootstrap_plan(
            self.truth, self.campaigns, resamples=25, seed=17
        )
        second = draw_valid_campaign_bootstrap_plan(
            self.truth, self.campaigns, resamples=25, seed=17
        )
        np.testing.assert_array_equal(first.draws, second.draws)
        macro, per_label = bootstrap_average_precision(
            self.truth, self.probability, first
        )
        self.assertEqual(macro.shape, (25,))
        self.assertEqual(per_label.shape, (25, 2))
        np.testing.assert_allclose(macro, 1.0)


if __name__ == "__main__":
    unittest.main()
