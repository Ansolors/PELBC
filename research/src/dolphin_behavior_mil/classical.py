"""Fold-local non-neural baselines for encounter-level multilabel prediction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from .modeling_data import DatasetRegistry, LABEL_ORDER, partition_sha256


def safe_float(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return float("nan")
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else float("nan")
    text = str(value).strip()
    if not text or text.casefold() in {"na", "n/a", "nan", "none", "unknown", "-"}:
        return float("nan")
    try:
        result = float(text)
    except ValueError:
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability))


def bag_size_records(
    registry: DatasetRegistry,
    encounter_ids: Iterable[str],
) -> list[dict[str, Any]]:
    records = []
    for encounter_id in registry.validate_encounter_ids(encounter_ids):
        durations = np.asarray(
            [float(row["duration_seconds"]) for row in registry.whistles[encounter_id]],
            dtype=np.float64,
        )
        records.append(
            {
                "encounter_id": encounter_id,
                "whistle_count": float(len(durations)),
                "log1p_whistle_count": float(np.log1p(len(durations))),
                "total_duration_seconds": float(np.sum(durations)),
                "mean_duration_seconds": float(np.mean(durations)),
                "std_duration_seconds": float(np.std(durations)),
                "minimum_duration_seconds": float(np.min(durations)),
                "duration_q25_seconds": _quantile(durations, 0.25),
                "median_duration_seconds": _quantile(durations, 0.50),
                "duration_q75_seconds": _quantile(durations, 0.75),
                "maximum_duration_seconds": float(np.max(durations)),
            }
        )
    return records


def metadata_records(
    registry: DatasetRegistry,
    encounter_ids: Iterable[str],
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    forbidden_source_prefixes: Sequence[str] = ("activity_",),
    forbidden_source_fields: Sequence[str] = (),
) -> list[dict[str, Any]]:
    requested = set(numeric_features) | set(categorical_features)
    forbidden = set(forbidden_source_fields)
    invalid = sorted(
        value
        for value in requested
        if value in forbidden or any(value.startswith(prefix) for prefix in forbidden_source_prefixes)
    )
    if invalid:
        raise ValueError(f"behavior-derived or forbidden metadata requested: {invalid}")

    records = []
    for encounter_id in registry.validate_encounter_ids(encounter_ids):
        encounter = registry.encounters[encounter_id]
        environment = registry.environments[encounter_id]
        derived = {
            "audio_year": str(encounter["audio_year"]),
            "audio_month": str(encounter["audio_date"])[5:7],
            "selected_location": encounter.get("selected_location"),
            "sample_rate_hz": str(encounter["sample_rate_hz"]),
        }
        record: dict[str, Any] = {"encounter_id": encounter_id}
        for feature in numeric_features:
            record[feature] = safe_float(environment.get(feature))
        for feature in categorical_features:
            value = derived.get(feature, environment.get(feature))
            record[feature] = None if value is None or not str(value).strip() else str(value).strip()
        records.append(record)
    return records


def _whistle_feature(row: Mapping[str, Any], name: str) -> float:
    if name in row:
        return safe_float(row.get(name))
    audit = row.get("audio_audit", {})
    if not isinstance(audit, Mapping):
        return float("nan")
    return safe_float(audit.get(name))


def handcrafted_records(
    registry: DatasetRegistry,
    encounter_ids: Iterable[str],
    *,
    per_whistle_features: Sequence[str],
    aggregation_statistics: Sequence[str],
    add_missing_fraction_per_feature: bool,
) -> list[dict[str, Any]]:
    allowed_statistics = {"mean", "std", "minimum", "q25", "median", "q75", "maximum"}
    unknown = sorted(set(aggregation_statistics) - allowed_statistics)
    if unknown:
        raise ValueError(f"unsupported aggregation statistics: {unknown}")
    records = []
    for encounter_id in registry.validate_encounter_ids(encounter_ids):
        whistles = registry.whistles[encounter_id]
        record: dict[str, Any] = {"encounter_id": encounter_id}
        for feature_name in per_whistle_features:
            values = np.asarray(
                [_whistle_feature(row, feature_name) for row in whistles], dtype=np.float64
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
                    record[key] = _quantile(finite, 0.25)
                elif statistic == "median":
                    record[key] = _quantile(finite, 0.50)
                elif statistic == "q75":
                    record[key] = _quantile(finite, 0.75)
                elif statistic == "maximum":
                    record[key] = float(np.max(finite))
            if add_missing_fraction_per_feature:
                record[f"{feature_name}__missing_fraction"] = float(
                    1.0 - len(finite) / len(values)
                )
        records.append(record)
    return records


@dataclass(frozen=True)
class FoldLocalTabularEncoder:
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    numeric_medians: np.ndarray
    numeric_means: np.ndarray
    numeric_standard_deviations: np.ndarray
    categories: Mapping[str, tuple[str, ...]]
    output_feature_names: tuple[str, ...]
    training_partition_sha256: str

    @classmethod
    def fit(
        cls,
        records: Sequence[Mapping[str, Any]],
        *,
        numeric_features: Sequence[str],
        categorical_features: Sequence[str] = (),
    ) -> "FoldLocalTabularEncoder":
        if not records:
            raise ValueError("tabular encoder requires training records")
        encounter_ids = [str(record["encounter_id"]) for record in records]
        if len(encounter_ids) != len(set(encounter_ids)):
            raise ValueError("tabular training records contain duplicate encounters")
        numeric = tuple(str(value) for value in numeric_features)
        categorical = tuple(str(value) for value in categorical_features)
        if not numeric and not categorical:
            raise ValueError("at least one tabular feature is required")
        if set(numeric) & set(categorical):
            raise ValueError("features cannot be both numeric and categorical")

        numeric_matrix = np.asarray(
            [[safe_float(record.get(name)) for name in numeric] for record in records],
            dtype=np.float64,
        )
        medians = np.zeros(len(numeric), dtype=np.float64)
        for index in range(len(numeric)):
            finite = numeric_matrix[np.isfinite(numeric_matrix[:, index]), index]
            medians[index] = float(np.median(finite)) if len(finite) else 0.0
        imputed = np.where(np.isfinite(numeric_matrix), numeric_matrix, medians[None, :])
        means = np.mean(imputed, axis=0) if len(numeric) else np.empty(0)
        standard_deviations = np.std(imputed, axis=0) if len(numeric) else np.empty(0)
        standard_deviations = np.where(standard_deviations < 1.0e-12, 1.0, standard_deviations)

        categories: dict[str, tuple[str, ...]] = {}
        for name in categorical:
            observed = {
                "__MISSING__" if record.get(name) is None else str(record[name])
                for record in records
            }
            observed.add("__MISSING__")
            categories[name] = tuple(sorted(observed))

        output_names = []
        for name in numeric:
            output_names.extend([name, f"{name}__missing"])
        for name in categorical:
            output_names.extend(f"{name}=={value}" for value in categories[name])
        return cls(
            numeric_features=numeric,
            categorical_features=categorical,
            numeric_medians=medians,
            numeric_means=np.asarray(means, dtype=np.float64),
            numeric_standard_deviations=np.asarray(standard_deviations, dtype=np.float64),
            categories=categories,
            output_feature_names=tuple(output_names),
            training_partition_sha256=partition_sha256(encounter_ids),
        )

    def transform(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        if not records:
            return np.empty((0, len(self.output_feature_names)), dtype=np.float64)
        columns = []
        for index, name in enumerate(self.numeric_features):
            values = np.asarray([safe_float(record.get(name)) for record in records])
            missing = ~np.isfinite(values)
            imputed = np.where(missing, self.numeric_medians[index], values)
            standardized = (
                imputed - self.numeric_means[index]
            ) / self.numeric_standard_deviations[index]
            columns.extend([standardized, missing.astype(np.float64)])
        for name in self.categorical_features:
            values = [
                "__MISSING__" if record.get(name) is None else str(record[name])
                for record in records
            ]
            for category in self.categories[name]:
                columns.append(np.asarray([value == category for value in values], dtype=np.float64))
        matrix = np.column_stack(columns)
        if matrix.shape[1] != len(self.output_feature_names) or not np.all(np.isfinite(matrix)):
            raise RuntimeError("tabular transformation produced an invalid matrix")
        return matrix

    def fit_transform(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        expected = partition_sha256(str(record["encounter_id"]) for record in records)
        if expected != self.training_partition_sha256:
            raise ValueError("fit_transform records differ from the fitted training partition")
        return self.transform(records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "encoder": "fold_local_tabular_v1",
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "numeric_medians": self.numeric_medians.tolist(),
            "numeric_means": self.numeric_means.tolist(),
            "numeric_standard_deviations": self.numeric_standard_deviations.tolist(),
            "categories": {key: list(value) for key, value in self.categories.items()},
            "output_feature_names": list(self.output_feature_names),
            "training_partition_sha256": self.training_partition_sha256,
        }


@dataclass(frozen=True)
class ClassicalFitResult:
    model_id: str
    encounter_ids: tuple[str, ...]
    campaign_ids: tuple[str, ...]
    labels: np.ndarray
    probabilities: np.ndarray
    tabular_encoder: Mapping[str, Any] | None
    candidate: Mapping[str, Any]
    model_state: Mapping[str, Any]


class PriorConstantBaseline:
    def __init__(self) -> None:
        self.prevalence_: np.ndarray | None = None

    def fit(self, y: np.ndarray) -> "PriorConstantBaseline":
        labels = np.asarray(y, dtype=np.float64)
        if labels.ndim != 2 or labels.shape[1] != len(LABEL_ORDER) or not len(labels):
            raise ValueError("labels must be a nonempty [encounter, 4] matrix")
        if not np.all((labels == 0) | (labels == 1)):
            raise ValueError("labels must be binary")
        self.prevalence_ = np.mean(labels, axis=0)
        return self

    def predict_proba(self, sample_count: int) -> np.ndarray:
        if self.prevalence_ is None:
            raise RuntimeError("baseline is not fitted")
        if sample_count < 0:
            raise ValueError("sample_count cannot be negative")
        return np.tile(self.prevalence_[None, :], (sample_count, 1))

    def to_dict(self) -> dict[str, Any]:
        if self.prevalence_ is None:
            raise RuntimeError("baseline is not fitted")
        return {
            "model": "training_label_prevalence",
            "prevalence": self.prevalence_.astype(float).tolist(),
        }


class IndependentLogisticBaseline:
    """One deterministic binary logistic head per behavior label."""

    def __init__(
        self,
        *,
        c_value: float,
        class_weight: str | None,
        random_seed: int,
        maximum_iterations: int = 5000,
    ) -> None:
        if c_value <= 0:
            raise ValueError("C must be positive")
        if class_weight not in {None, "balanced"}:
            raise ValueError("class_weight must be None or balanced")
        self.c_value = float(c_value)
        self.class_weight = class_weight
        self.random_seed = int(random_seed)
        self.maximum_iterations = int(maximum_iterations)
        self.heads_: list[LogisticRegression | float] | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "IndependentLogisticBaseline":
        features = np.asarray(x, dtype=np.float64)
        labels = np.asarray(y, dtype=np.int8)
        if features.ndim != 2 or not len(features) or not np.all(np.isfinite(features)):
            raise ValueError("features must be a finite nonempty matrix")
        if labels.shape != (len(features), len(LABEL_ORDER)):
            raise ValueError("labels must match features and contain four columns")
        if not np.all((labels == 0) | (labels == 1)):
            raise ValueError("labels must be binary")
        heads: list[LogisticRegression | float] = []
        for label_index in range(labels.shape[1]):
            target = labels[:, label_index]
            unique = np.unique(target)
            if len(unique) == 1:
                heads.append(float(unique[0]))
                continue
            head = LogisticRegression(
                C=self.c_value,
                penalty="l2",
                solver="liblinear",
                class_weight=self.class_weight,
                max_iter=self.maximum_iterations,
                random_state=self.random_seed + label_index,
            )
            head.fit(features, target)
            heads.append(head)
        self.heads_ = heads
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        features = np.asarray(x, dtype=np.float64)
        if features.ndim != 2 or not np.all(np.isfinite(features)):
            raise ValueError("features must be a finite matrix")
        if self.heads_ is None:
            raise RuntimeError("baseline is not fitted")
        columns = []
        for head in self.heads_:
            if isinstance(head, float):
                columns.append(np.full(len(features), head, dtype=np.float64))
            else:
                columns.append(head.predict_proba(features)[:, 1])
        result = np.column_stack(columns)
        return np.clip(result, 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        if self.heads_ is None:
            raise RuntimeError("baseline is not fitted")
        heads = []
        for head in self.heads_:
            if isinstance(head, float):
                heads.append({"kind": "constant", "probability": head})
            else:
                heads.append(
                    {
                        "kind": "logistic_regression",
                        "classes": head.classes_.astype(int).tolist(),
                        "coefficient": head.coef_.astype(float).tolist(),
                        "intercept": head.intercept_.astype(float).tolist(),
                        "iterations": head.n_iter_.astype(int).tolist(),
                    }
                )
        return {
            "model": "independent_l2_logistic_heads",
            "c_value": self.c_value,
            "class_weight": self.class_weight,
            "random_seed": self.random_seed,
            "maximum_iterations": self.maximum_iterations,
            "heads": heads,
        }


def label_matrix(registry: DatasetRegistry, encounter_ids: Iterable[str]) -> np.ndarray:
    ids = registry.validate_encounter_ids(encounter_ids)
    return np.stack([registry.labels[value] for value in ids]).astype(np.float32)


def classical_candidate_grid(
    c_values: Sequence[float],
    class_weight_options: Sequence[str],
) -> list[dict[str, Any]]:
    candidates = []
    for c_value in c_values:
        for class_weight in class_weight_options:
            normalized = None if str(class_weight) == "none" else str(class_weight)
            if normalized not in {None, "balanced"}:
                raise ValueError(f"unsupported class-weight option {class_weight}")
            candidates.append(
                {
                    "candidate_id": f"c{len(candidates) + 1:02d}",
                    "c_value": float(c_value),
                    "class_weight": normalized,
                }
            )
    return candidates


def fit_classical_baseline(
    registry: DatasetRegistry,
    train_encounter_ids: Iterable[str],
    validation_encounter_ids: Iterable[str],
    *,
    model_id: str,
    modeling_config: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    random_seed: int,
) -> ClassicalFitResult:
    train_ids = registry.validate_encounter_ids(train_encounter_ids)
    validation_ids = registry.validate_encounter_ids(validation_encounter_ids)
    registry.assert_campaign_disjoint(train_ids, validation_ids)
    supported = {
        "prior_constant",
        "bag_size_logistic",
        "metadata_logistic",
        "handcrafted_logistic",
    }
    if model_id not in supported:
        raise ValueError(f"unsupported classical baseline {model_id}")
    train_y = label_matrix(registry, train_ids)
    validation_y = label_matrix(registry, validation_ids)
    if model_id == "prior_constant":
        if candidate is not None and bool(candidate):
            raise ValueError("prior baseline has no hyperparameter candidate")
        model = PriorConstantBaseline().fit(train_y)
        probability = model.predict_proba(len(validation_ids))
        return ClassicalFitResult(
            model_id=model_id,
            encounter_ids=validation_ids,
            campaign_ids=tuple(registry.campaign_by_encounter[value] for value in validation_ids),
            labels=validation_y.astype(np.int8),
            probabilities=probability,
            tabular_encoder=None,
            candidate={"candidate_id": "prior_only"},
            model_state=model.to_dict(),
        )

    if not candidate:
        raise ValueError("a logistic candidate is required")
    if model_id == "bag_size_logistic":
        section = modeling_config["bag_size_logistic"]
        numeric_names = tuple(section["feature_names"])
        categorical_names: tuple[str, ...] = ()
        train_records = bag_size_records(registry, train_ids)
        validation_records = bag_size_records(registry, validation_ids)
    elif model_id == "metadata_logistic":
        section = modeling_config["metadata_logistic"]
        numeric_names = tuple(section["numeric_features"])
        categorical_names = tuple(section["categorical_features"])
        keyword = {
            "numeric_features": numeric_names,
            "categorical_features": categorical_names,
            "forbidden_source_prefixes": tuple(section["forbidden_source_prefixes"]),
            "forbidden_source_fields": tuple(section["forbidden_source_fields"]),
        }
        train_records = metadata_records(registry, train_ids, **keyword)
        validation_records = metadata_records(registry, validation_ids, **keyword)
    else:
        section = modeling_config["handcrafted_logistic"]
        keyword = {
            "per_whistle_features": tuple(section["per_whistle_features"]),
            "aggregation_statistics": tuple(section["aggregation_statistics"]),
            "add_missing_fraction_per_feature": bool(
                section["add_missing_fraction_per_feature"]
            ),
        }
        train_records = handcrafted_records(registry, train_ids, **keyword)
        validation_records = handcrafted_records(registry, validation_ids, **keyword)
        numeric_names = tuple(sorted(set(train_records[0]) - {"encounter_id"}))
        categorical_names = ()

    encoder = FoldLocalTabularEncoder.fit(
        train_records,
        numeric_features=numeric_names,
        categorical_features=categorical_names,
    )
    train_x = encoder.fit_transform(train_records)
    validation_x = encoder.transform(validation_records)
    model = IndependentLogisticBaseline(
        c_value=float(candidate["c_value"]),
        class_weight=candidate.get("class_weight"),
        random_seed=int(random_seed),
        maximum_iterations=int(modeling_config["classical"]["common"]["maximum_iterations"]),
    ).fit(train_x, train_y)
    probability = model.predict_proba(validation_x)
    return ClassicalFitResult(
        model_id=model_id,
        encounter_ids=validation_ids,
        campaign_ids=tuple(registry.campaign_by_encounter[value] for value in validation_ids),
        labels=validation_y.astype(np.int8),
        probabilities=probability,
        tabular_encoder=encoder.to_dict(),
        candidate=dict(candidate),
        model_state=model.to_dict(),
    )
