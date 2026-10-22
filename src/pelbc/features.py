"""Encounter aggregation and fold-compatible tabular encoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np


PER_WHISTLE_FEATURES = (
    "duration_seconds",
    "core_tonal_component_frequency_q05_hz",
    "core_tonal_component_frequency_q50_hz",
    "core_tonal_component_frequency_q95_hz",
    "core_tonal_component_frequency_max_hz",
    "core_tonal_component_bandwidth_q90_hz",
    "core_tonal_component_duration_seconds",
    "core_tonal_component_frame_fraction",
    "core_tonal_component_median_salience_db",
    "core_tonal_component_peak_salience_db",
    "rms_dbfs",
    "temporal_dynamic_db",
    "exact_clipping_fraction",
    "low_activity_frame_fraction",
)

AGGREGATION_STATISTICS = (
    "mean",
    "std",
    "minimum",
    "q25",
    "median",
    "q75",
    "maximum",
)


def safe_float(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return float("nan")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def aggregate_whistles(
    clip_features: Sequence[Mapping[str, Any]],
    *,
    per_whistle_features: Sequence[str] = PER_WHISTLE_FEATURES,
    aggregation_statistics: Sequence[str] = AGGREGATION_STATISTICS,
    add_missing_fraction_per_feature: bool = True,
) -> dict[str, float]:
    """Aggregate a variable-size whistle bag exactly as in Acoustic LR."""

    if not clip_features:
        raise ValueError("at least one whistle feature record is required")
    allowed = set(AGGREGATION_STATISTICS)
    unknown = sorted(set(aggregation_statistics) - allowed)
    if unknown:
        raise ValueError(f"unsupported aggregation statistics: {unknown}")

    record: dict[str, float] = {}
    for feature_name in per_whistle_features:
        values = np.asarray(
            [safe_float(row.get(feature_name)) for row in clip_features],
            dtype=np.float64,
        )
        finite = values[np.isfinite(values)]
        for statistic in aggregation_statistics:
            key = f"{feature_name}__{statistic}"
            if not len(finite):
                record[key] = float("nan")
            elif statistic == "mean":
                record[key] = float(np.mean(finite))
            elif statistic == "std":
                record[key] = float(np.std(finite))
            elif statistic == "minimum":
                record[key] = float(np.min(finite))
            elif statistic == "q25":
                record[key] = float(np.quantile(finite, 0.25))
            elif statistic == "median":
                record[key] = float(np.quantile(finite, 0.50))
            elif statistic == "q75":
                record[key] = float(np.quantile(finite, 0.75))
            elif statistic == "maximum":
                record[key] = float(np.max(finite))
        if add_missing_fraction_per_feature:
            record[f"{feature_name}__missing_fraction"] = float(
                1.0 - len(finite) / len(values)
            )
    return record


def encode_aggregates(
    record: Mapping[str, Any],
    encoder: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply serialized median imputation, scaling, and missing indicators."""

    names = tuple(str(value) for value in encoder["numeric_features"])
    medians = np.asarray(encoder["numeric_medians"], dtype=np.float64)
    means = np.asarray(encoder["numeric_means"], dtype=np.float64)
    deviations = np.asarray(
        encoder["numeric_standard_deviations"], dtype=np.float64
    )
    if not (
        len(names) == len(medians) == len(means) == len(deviations)
        and np.all(np.isfinite(medians))
        and np.all(np.isfinite(means))
        and np.all(np.isfinite(deviations))
        and np.all(deviations > 0)
    ):
        raise ValueError("serialized numeric encoder is inconsistent")

    columns: list[float] = []
    standardized_by_feature: dict[str, float] = {}
    output_names: list[str] = []
    for index, name in enumerate(names):
        value = safe_float(record.get(name))
        missing = not math.isfinite(value)
        imputed = medians[index] if missing else value
        standardized = float((imputed - means[index]) / deviations[index])
        columns.extend([standardized, float(missing)])
        output_names.extend([name, f"{name}__missing"])
        standardized_by_feature[name] = standardized

    expected_output_names = tuple(str(value) for value in encoder["output_feature_names"])
    if tuple(output_names) != expected_output_names:
        raise ValueError("serialized encoder output feature order is inconsistent")
    matrix = np.asarray(columns, dtype=np.float64)
    if matrix.ndim != 1 or not np.all(np.isfinite(matrix)):
        raise ValueError("tabular encoding produced invalid values")
    return matrix, standardized_by_feature
