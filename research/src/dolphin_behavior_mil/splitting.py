"""Deterministic campaign grouping and balanced grouped partitioning."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


def canonical_location(value: object) -> str:
    """Normalize location codes conservatively by whitespace and case only."""

    if value is None:
        return "missing"
    text = " ".join(str(value).strip().split()).casefold()
    return text or "missing"


def campaign_ids_by_date(
    dates: Iterable[str],
    *,
    maximum_gap_days: int = 1,
) -> dict[str, str]:
    """Group maximal within-year chains of consecutive observed recording dates."""

    if maximum_gap_days < 0:
        raise ValueError("maximum_gap_days must be nonnegative")
    parsed = sorted({date.fromisoformat(str(value)) for value in dates})
    if not parsed:
        return {}
    blocks: list[list[date]] = []
    current: list[date] = []
    for value in parsed:
        if current:
            gap = (value - current[-1]).days
            if value.year != current[-1].year or gap > maximum_gap_days:
                blocks.append(current)
                current = []
        current.append(value)
    blocks.append(current)

    output: dict[str, str] = {}
    for block in blocks:
        start = block[0].strftime("%Y%m%d")
        end = block[-1].strftime("%Y%m%d")
        campaign_id = f"camp_{start}" if start == end else f"camp_{start}_{end}"
        for value in block:
            output[value.isoformat()] = campaign_id
    return output


def equal_capacities(item_count: int, fold_count: int) -> tuple[int, ...]:
    if item_count < fold_count or fold_count < 2:
        raise ValueError("need at least one item per fold and at least two folds")
    quotient, remainder = divmod(item_count, fold_count)
    return tuple(
        quotient + (1 if index < remainder else 0) for index in range(fold_count)
    )


def partition_objective(
    assignments: np.ndarray,
    features: np.ndarray,
    weights: np.ndarray,
    fold_count: int,
) -> float:
    """Return normalized weighted squared imbalance across folds."""

    if features.ndim != 2 or assignments.ndim != 1:
        raise ValueError("invalid feature or assignment dimensions")
    if len(assignments) != len(features) or len(weights) != features.shape[1]:
        raise ValueError("feature, weight and assignment sizes disagree")
    fold_sums = np.zeros((fold_count, features.shape[1]), dtype=np.float64)
    for fold in range(fold_count):
        fold_sums[fold] = np.sum(features[assignments == fold], axis=0)
    target = np.sum(features, axis=0) / fold_count
    scale = np.maximum(np.abs(target), 1.0)
    normalized = (fold_sums - target[None, :]) / scale[None, :]
    return float(np.mean(np.square(normalized) * weights[None, :]))


def _canonicalize_assignments(
    assignments: np.ndarray,
    group_ids: Sequence[str],
) -> np.ndarray:
    folds = sorted(
        set(int(value) for value in assignments),
        key=lambda fold: tuple(
            sorted(
                group_ids[index]
                for index, value in enumerate(assignments)
                if int(value) == fold
            )
        ),
    )
    remap = {old: new for new, old in enumerate(folds)}
    return np.asarray([remap[int(value)] for value in assignments], dtype=np.int64)


def optimize_group_partition(
    group_ids: Sequence[str],
    features: np.ndarray,
    weights: Sequence[float],
    *,
    fold_count: int,
    seed: int,
    restarts: int = 48,
    iterations_per_restart: int = 4000,
) -> tuple[dict[str, int], float]:
    """Optimize a reproducible group-disjoint fold assignment by swap search.

    Fold group counts are constrained to differ by at most one. Only aggregate
    labels and nuisance summaries supplied in ``features`` affect the assignment;
    no acoustic feature values are inspected.
    """

    group_ids = tuple(str(value) for value in group_ids)
    feature_array = np.asarray(features, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("group IDs must be unique")
    if len(group_ids) != len(feature_array):
        raise ValueError("one feature row is required per group")
    if restarts < 1 or iterations_per_restart < 1:
        raise ValueError("search budget must be positive")
    if not np.all(np.isfinite(feature_array)) or np.any(feature_array < 0):
        raise ValueError("partition features must be finite nonnegative values")
    if not np.all(np.isfinite(weight_array)) or np.any(weight_array <= 0):
        raise ValueError("partition weights must be finite positive values")

    capacities = equal_capacities(len(group_ids), fold_count)
    base = np.concatenate(
        [np.full(capacity, fold, dtype=np.int64) for fold, capacity in enumerate(capacities)]
    )
    target = np.sum(feature_array, axis=0) / fold_count
    scale = np.maximum(np.abs(target), 1.0)

    def objective_from_sums(fold_sums: np.ndarray) -> float:
        normalized = (fold_sums - target[None, :]) / scale[None, :]
        return float(np.mean(np.square(normalized) * weight_array[None, :]))

    best_assignment: np.ndarray | None = None
    best_score = math.inf
    rng = np.random.default_rng(int(seed))
    for restart in range(restarts):
        assignments = base[rng.permutation(len(base))].copy()
        fold_sums = np.zeros((fold_count, feature_array.shape[1]), dtype=np.float64)
        for fold in range(fold_count):
            fold_sums[fold] = np.sum(feature_array[assignments == fold], axis=0)
        score = objective_from_sums(fold_sums)
        local_best_score = score
        local_best = assignments.copy()
        start_temperature = max(score * 0.05, 1.0e-5)
        end_temperature = max(start_temperature * 0.002, 1.0e-9)
        for iteration in range(iterations_per_restart):
            first, second = rng.integers(0, len(assignments), size=2)
            if first == second or assignments[first] == assignments[second]:
                continue
            first_fold = int(assignments[first])
            second_fold = int(assignments[second])
            candidate_sums = fold_sums.copy()
            candidate_sums[first_fold] += feature_array[second] - feature_array[first]
            candidate_sums[second_fold] += feature_array[first] - feature_array[second]
            candidate_score = objective_from_sums(candidate_sums)
            progress = iteration / max(iterations_per_restart - 1, 1)
            temperature = start_temperature * math.pow(
                end_temperature / start_temperature,
                progress,
            )
            delta = candidate_score - score
            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                assignments[first], assignments[second] = (
                    assignments[second],
                    assignments[first],
                )
                fold_sums = candidate_sums
                score = candidate_score
                if score < local_best_score:
                    local_best_score = score
                    local_best = assignments.copy()
        if local_best_score < best_score:
            best_score = local_best_score
            best_assignment = local_best

    if best_assignment is None:  # pragma: no cover - guarded by positive restarts
        raise RuntimeError("partition search failed")
    best_assignment = _canonicalize_assignments(best_assignment, group_ids)
    best_score = partition_objective(
        best_assignment,
        feature_array,
        weight_array,
        fold_count,
    )
    return {
        group_id: int(best_assignment[index])
        for index, group_id in enumerate(group_ids)
    }, best_score


def grouped_membership_is_disjoint(
    assignments: Mapping[str, int],
    members_by_group: Mapping[str, Sequence[str]],
) -> bool:
    """Check that each member occurs once and every assigned group exists."""

    if set(assignments) != set(members_by_group):
        return False
    seen: set[str] = set()
    for group_id in assignments:
        members = {str(value) for value in members_by_group[group_id]}
        if seen.intersection(members):
            return False
        seen.update(members)
    return True


def group_values(
    rows: Sequence[Mapping[str, object]],
    group_key: str,
) -> dict[str, list[Mapping[str, object]]]:
    output: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        output[str(row[group_key])].append(row)
    return dict(output)
