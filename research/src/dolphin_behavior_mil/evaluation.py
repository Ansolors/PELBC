"""Frozen encounter-level metrics and campaign-cluster inference utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score


def validate_multilabel_arrays(
    y_true: np.ndarray,
    y_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true)
    probability = np.asarray(y_probability, dtype=np.float64)
    if truth.ndim != 2 or probability.shape != truth.shape:
        raise ValueError("truth and probability must be matching 2D arrays")
    if not np.all((truth == 0) | (truth == 1)):
        raise ValueError("truth must be binary")
    if not np.all(np.isfinite(probability)) or np.any(
        (probability < 0) | (probability > 1)
    ):
        raise ValueError("probabilities must be finite values in [0, 1]")
    return truth.astype(np.int8, copy=False), probability


def per_label_average_precision(
    y_true: np.ndarray,
    y_probability: np.ndarray,
) -> np.ndarray:
    truth, probability = validate_multilabel_arrays(y_true, y_probability)
    if np.any(np.sum(truth, axis=0) == 0):
        raise ValueError("average precision requires a positive for every label")
    return np.asarray(
        [
            average_precision_score(truth[:, index], probability[:, index])
            for index in range(truth.shape[1])
        ],
        dtype=np.float64,
    )


def macro_average_precision(
    y_true: np.ndarray,
    y_probability: np.ndarray,
) -> float:
    return float(np.mean(per_label_average_precision(y_true, y_probability)))


def f1_at_threshold(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float,
) -> float:
    truth = np.asarray(y_true, dtype=np.int8)
    probability = np.asarray(y_probability, dtype=np.float64)
    if truth.ndim != 1 or probability.shape != truth.shape:
        raise ValueError("binary truth and probability must be matching vectors")
    predicted = probability >= float(threshold)
    true_positive = int(np.sum((truth == 1) & predicted))
    false_positive = int(np.sum((truth == 0) & predicted))
    false_negative = int(np.sum((truth == 1) & ~predicted))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def select_largest_max_f1_threshold(
    y_true: np.ndarray,
    y_probability: np.ndarray,
) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=np.int8)
    probability = np.asarray(y_probability, dtype=np.float64)
    if truth.ndim != 1 or probability.shape != truth.shape:
        raise ValueError("binary truth and probability must be matching vectors")
    if not np.all((truth == 0) | (truth == 1)):
        raise ValueError("truth must be binary")
    if not np.all(np.isfinite(probability)) or np.any(
        (probability < 0) | (probability > 1)
    ):
        raise ValueError("probabilities must lie in [0, 1]")
    candidates = np.unique(np.concatenate(([0.0, 1.0], probability)))
    scores = np.asarray(
        [f1_at_threshold(truth, probability, value) for value in candidates]
    )
    maximum = float(np.max(scores))
    tied = candidates[np.isclose(scores, maximum, rtol=0.0, atol=1.0e-15)]
    return {"threshold": float(np.max(tied)), "f1": maximum}


def adaptive_ece(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    *,
    bin_count: int = 10,
) -> float:
    truth = np.asarray(y_true, dtype=np.float64)
    probability = np.asarray(y_probability, dtype=np.float64)
    if truth.ndim != 1 or probability.shape != truth.shape or not len(truth):
        raise ValueError("truth and probability must be nonempty matching vectors")
    if bin_count < 1:
        raise ValueError("bin_count must be positive")
    order = np.argsort(probability, kind="mergesort")
    bins = np.array_split(order, min(bin_count, len(order)))
    error = 0.0
    for indices in bins:
        error += len(indices) / len(order) * abs(
            float(np.mean(truth[indices])) - float(np.mean(probability[indices]))
        )
    return float(error)


def _cluster_indices(cluster_ids: Sequence[object]) -> tuple[list[str], dict[str, np.ndarray]]:
    values = np.asarray([str(value) for value in cluster_ids], dtype=object)
    groups = sorted(set(values.tolist()))
    return groups, {group: np.flatnonzero(values == group) for group in groups}


def campaign_cluster_bootstrap(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    campaign_ids: Sequence[object],
    *,
    resamples: int,
    seed: int,
    maximum_attempt_multiplier: int = 100,
) -> np.ndarray:
    """Bootstrap complete campaign blocks and return macro-AP replicates."""

    truth, probability = validate_multilabel_arrays(y_true, y_probability)
    if len(campaign_ids) != len(truth):
        raise ValueError("one campaign ID is required per encounter")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    groups, indices_by_group = _cluster_indices(campaign_ids)
    if len(groups) < 2:
        raise ValueError("at least two campaign clusters are required")
    rng = np.random.default_rng(int(seed))
    output: list[float] = []
    maximum_attempts = resamples * maximum_attempt_multiplier
    attempts = 0
    while len(output) < resamples and attempts < maximum_attempts:
        attempts += 1
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([indices_by_group[str(group)] for group in sampled])
        sampled_truth = truth[indices]
        positives = np.sum(sampled_truth, axis=0)
        if np.any(positives == 0) or np.any(positives == len(sampled_truth)):
            continue
        output.append(macro_average_precision(sampled_truth, probability[indices]))
    if len(output) != resamples:
        raise RuntimeError("could not draw enough valid cluster bootstrap replicates")
    return np.asarray(output, dtype=np.float64)


def percentile_interval(
    values: Sequence[float],
    *,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("values must be a finite nonempty vector")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence level must lie strictly between zero and one")
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(array, [tail, 1.0 - tail])
    return float(lower), float(upper)


@dataclass(frozen=True)
class PairedPermutationResult:
    observed_delta: float
    p_value: float
    permutations: int
    exceedance_count: int


def paired_campaign_prediction_swap_test(
    y_true: np.ndarray,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    campaign_ids: Sequence[object],
    *,
    permutations: int,
    seed: int,
) -> PairedPermutationResult:
    """Swap complete campaign prediction blocks between two paired models."""

    truth, first = validate_multilabel_arrays(y_true, probability_a)
    _, second = validate_multilabel_arrays(y_true, probability_b)
    if len(campaign_ids) != len(truth):
        raise ValueError("one campaign ID is required per encounter")
    if permutations < 1:
        raise ValueError("permutations must be positive")
    groups, indices_by_group = _cluster_indices(campaign_ids)
    observed = macro_average_precision(truth, first) - macro_average_precision(
        truth, second
    )
    rng = np.random.default_rng(int(seed))
    exceedances = 0
    for _ in range(permutations):
        permuted_first = first.copy()
        permuted_second = second.copy()
        swap_flags = rng.integers(0, 2, size=len(groups), dtype=np.int8)
        for group, swap in zip(groups, swap_flags, strict=True):
            if not swap:
                continue
            indices = indices_by_group[group]
            permuted_first[indices] = second[indices]
            permuted_second[indices] = first[indices]
        delta = macro_average_precision(truth, permuted_first) - macro_average_precision(
            truth, permuted_second
        )
        exceedances += abs(delta) >= abs(observed) - 1.0e-15
    p_value = (1 + exceedances) / (permutations + 1)
    return PairedPermutationResult(
        observed_delta=float(observed),
        p_value=float(p_value),
        permutations=int(permutations),
        exceedance_count=int(exceedances),
    )


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    if not p_values:
        return {}
    if any(not 0 <= float(value) <= 1 for value in p_values.values()):
        raise ValueError("p-values must lie in [0, 1]")
    ordered = sorted(p_values, key=lambda key: (float(p_values[key]), str(key)))
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        candidate = min(1.0, (count - rank) * float(p_values[key]))
        running = max(running, candidate)
        adjusted[key] = running
    return {key: adjusted[key] for key in p_values}


def evaluation_record(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    label_names: Sequence[str],
) -> dict[str, Any]:
    truth, probability = validate_multilabel_arrays(y_true, y_probability)
    if len(label_names) != truth.shape[1]:
        raise ValueError("one label name is required per output column")
    per_label = per_label_average_precision(truth, probability)
    return {
        "encounter_count": int(len(truth)),
        "macro_average_precision": float(np.mean(per_label)),
        "per_label_average_precision": {
            str(label): float(value)
            for label, value in zip(label_names, per_label, strict=True)
        },
        "positive_counts": {
            str(label): int(value)
            for label, value in zip(label_names, np.sum(truth, axis=0), strict=True)
        },
    }
