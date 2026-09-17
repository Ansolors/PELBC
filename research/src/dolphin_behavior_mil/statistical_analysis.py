"""Publication-level metrics and campaign-cluster resampling helpers.

The independent analysis unit is always an encounter.  Campaign blocks are used
only for uncertainty estimation so that encounters recorded in the same short
field campaign are never resampled as if they were independent observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .evaluation import adaptive_ece, validate_multilabel_arrays


def _label_names(label_names: Sequence[str], width: int) -> tuple[str, ...]:
    names = tuple(str(value) for value in label_names)
    if len(names) != width or len(set(names)) != width:
        raise ValueError("label names must be unique and match the output width")
    return names


def probability_metrics(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    label_names: Sequence[str],
    *,
    roc_auc_minimum_positive: int = 3,
    roc_auc_minimum_negative: int = 3,
    ece_bin_count: int = 10,
) -> dict[str, Any]:
    """Return threshold-free discrimination and calibration metrics."""

    truth, probability = validate_multilabel_arrays(y_true, y_probability)
    names = _label_names(label_names, truth.shape[1])
    if roc_auc_minimum_positive < 1 or roc_auc_minimum_negative < 1:
        raise ValueError("ROC-AUC class-count minima must be positive")

    positives = np.sum(truth, axis=0).astype(int)
    negatives = (len(truth) - positives).astype(int)
    average_precision_values: list[float | None] = []
    average_precision_records: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(names):
        positive_count = int(positives[index])
        if positive_count == 0:
            value = None
            reason = "no positive encounters in the evaluated subset"
        else:
            value = float(
                average_precision_score(truth[:, index], probability[:, index])
            )
            reason = None
        average_precision_values.append(value)
        average_precision_records[name] = {
            "value": value,
            "defined": value is not None,
            "positive_count": positive_count,
            "negative_count": int(negatives[index]),
            "undefined_reason": reason,
        }
    brier = np.mean((probability - truth) ** 2, axis=0)
    ece = np.asarray(
        [
            adaptive_ece(
                truth[:, index], probability[:, index], bin_count=ece_bin_count
            )
            for index in range(truth.shape[1])
        ],
        dtype=np.float64,
    )

    roc_values: list[float | None] = []
    roc_records: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(names):
        positive_count = int(positives[index])
        negative_count = int(negatives[index])
        if (
            positive_count < roc_auc_minimum_positive
            or negative_count < roc_auc_minimum_negative
        ):
            value = None
            reason = (
                "fewer than the prespecified minimum positive or negative "
                "encounters"
            )
        else:
            value = float(roc_auc_score(truth[:, index], probability[:, index]))
            reason = None
            roc_values.append(value)
        roc_records[name] = {
            "value": value,
            "defined": value is not None,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "undefined_reason": reason,
        }

    return {
        "encounter_count": int(len(truth)),
        "positive_counts": {
            name: int(value) for name, value in zip(names, positives, strict=True)
        },
        "negative_counts": {
            name: int(value) for name, value in zip(names, negatives, strict=True)
        },
        "macro_average_precision": (
            float(np.mean([value for value in average_precision_values if value is not None]))
            if all(value is not None for value in average_precision_values)
            else None
        ),
        "per_label_average_precision": {
            name: float(value)
            if value is not None
            else None
            for name, value in zip(names, average_precision_values, strict=True)
        },
        "per_label_average_precision_status": average_precision_records,
        "macro_roc_auc": (
            float(np.mean(roc_values)) if len(roc_values) == len(names) else None
        ),
        "per_label_roc_auc": roc_records,
        "macro_brier_score": float(np.mean(brier)),
        "per_label_brier_score": {
            name: float(value) for name, value in zip(names, brier, strict=True)
        },
        "macro_adaptive_ece": float(np.mean(ece)),
        "per_label_adaptive_ece": {
            name: float(value) for name, value in zip(names, ece, strict=True)
        },
        "adaptive_ece_bin_count": int(ece_bin_count),
    }


def decision_metrics(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    thresholds: float | Sequence[float] | np.ndarray,
    label_names: Sequence[str],
) -> dict[str, Any]:
    """Return pooled multilabel metrics for fixed or row-specific thresholds."""

    truth, probability = validate_multilabel_arrays(y_true, y_probability)
    names = _label_names(label_names, truth.shape[1])
    threshold_array = np.asarray(thresholds, dtype=np.float64)
    if threshold_array.ndim == 0:
        threshold_array = np.full(probability.shape, float(threshold_array))
    elif threshold_array.shape == (truth.shape[1],):
        threshold_array = np.broadcast_to(threshold_array, probability.shape)
    elif threshold_array.shape != probability.shape:
        raise ValueError(
            "thresholds must be scalar, one value per label, or one matrix value "
            "per encounter and label"
        )
    if not np.all(np.isfinite(threshold_array)) or np.any(
        (threshold_array < 0) | (threshold_array > 1)
    ):
        raise ValueError("thresholds must be finite values in [0, 1]")

    predicted = probability >= threshold_array
    true_positive = np.sum((truth == 1) & predicted, axis=0)
    false_positive = np.sum((truth == 0) & predicted, axis=0)
    false_negative = np.sum((truth == 1) & ~predicted, axis=0)
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator != 0,
    )
    return {
        "encounter_count": int(len(truth)),
        "macro_f1": float(np.mean(f1)),
        "per_label_f1": {
            name: float(value) for name, value in zip(names, f1, strict=True)
        },
        "hamming_loss": float(np.mean(predicted != truth)),
        "predicted_positive_counts": {
            name: int(value)
            for name, value in zip(names, np.sum(predicted, axis=0), strict=True)
        },
    }


@dataclass(frozen=True)
class CampaignBootstrapPlan:
    """A reusable set of valid campaign-block bootstrap draws."""

    campaign_order: tuple[str, ...]
    indices_by_campaign: tuple[np.ndarray, ...]
    draws: np.ndarray
    seed: int
    attempts: int

    @property
    def resamples(self) -> int:
        return int(self.draws.shape[0])

    def encounter_indices(self, replicate_index: int) -> np.ndarray:
        draw = self.draws[int(replicate_index)]
        return np.concatenate(
            [self.indices_by_campaign[int(group_index)] for group_index in draw]
        )


def draw_valid_campaign_bootstrap_plan(
    y_true: np.ndarray,
    campaign_ids: Sequence[object],
    *,
    resamples: int,
    seed: int,
    maximum_attempt_multiplier: int = 100,
) -> CampaignBootstrapPlan:
    """Draw complete campaigns, redrawing class-degenerate replicates."""

    truth = np.asarray(y_true)
    if truth.ndim != 2 or not np.all((truth == 0) | (truth == 1)):
        raise ValueError("truth must be a binary 2D array")
    if len(campaign_ids) != len(truth):
        raise ValueError("one campaign ID is required per encounter")
    if resamples < 1 or maximum_attempt_multiplier < 1:
        raise ValueError("resample and attempt counts must be positive")
    values = np.asarray([str(value) for value in campaign_ids], dtype=object)
    campaigns = tuple(sorted(set(values.tolist())))
    if len(campaigns) < 2:
        raise ValueError("at least two campaign clusters are required")
    grouped = tuple(np.flatnonzero(values == campaign) for campaign in campaigns)
    rng = np.random.default_rng(int(seed))
    accepted: list[np.ndarray] = []
    attempts = 0
    maximum_attempts = int(resamples) * int(maximum_attempt_multiplier)
    while len(accepted) < int(resamples) and attempts < maximum_attempts:
        attempts += 1
        draw = rng.integers(0, len(campaigns), size=len(campaigns), dtype=np.int16)
        indices = np.concatenate([grouped[int(index)] for index in draw])
        sampled_truth = truth[indices]
        positives = np.sum(sampled_truth, axis=0)
        if np.any(positives == 0) or np.any(positives == len(sampled_truth)):
            continue
        accepted.append(draw)
    if len(accepted) != int(resamples):
        raise RuntimeError("could not draw enough valid campaign bootstrap replicates")
    return CampaignBootstrapPlan(
        campaign_order=campaigns,
        indices_by_campaign=grouped,
        draws=np.stack(accepted),
        seed=int(seed),
        attempts=int(attempts),
    )


def bootstrap_average_precision(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    plan: CampaignBootstrapPlan,
) -> tuple[np.ndarray, np.ndarray]:
    """Return macro and per-label AP for every draw in a frozen plan."""

    truth, probability = validate_multilabel_arrays(y_true, y_probability)
    macro = np.empty(plan.resamples, dtype=np.float64)
    per_label = np.empty((plan.resamples, truth.shape[1]), dtype=np.float64)
    for replicate in range(plan.resamples):
        indices = plan.encounter_indices(replicate)
        values = np.asarray(
            [
                average_precision_score(
                    truth[indices, label_index],
                    probability[indices, label_index],
                )
                for label_index in range(truth.shape[1])
            ],
            dtype=np.float64,
        )
        per_label[replicate] = values
        macro[replicate] = float(np.mean(values))
    return macro, per_label
