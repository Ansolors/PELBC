"""Selection, calibration and OOF assembly for grouped nested validation."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from .classical import classical_candidate_grid
from .evaluation import macro_average_precision, select_largest_max_f1_threshold
from .modeling_data import LABEL_ORDER


CLASSICAL_MODEL_IDS = {
    "prior_constant",
    "bag_size_logistic",
    "metadata_logistic",
    "handcrafted_logistic",
}
CNN_MODEL_IDS = {
    "cnn_mean",
    "cnn_max",
    "cnn_linear_softmax",
    "cnn_shared_gated_attention",
    "hf_lw_gam",
}
PRETRAINED_MODEL_IDS = {"panns_frozen_mil", "aves_frozen_mil"}


def candidates_for_model(
    modeling_config: Mapping[str, Any],
    model_id: str,
) -> list[dict[str, Any]]:
    if model_id == "prior_constant":
        return [{"candidate_id": "prior_only"}]
    if model_id in CLASSICAL_MODEL_IDS:
        search = modeling_config["classical"]["search"]
        return classical_candidate_grid(
            search["c_values"], search["class_weight_options"]
        )
    if model_id in CNN_MODEL_IDS:
        return [dict(value) for value in modeling_config["neural_candidates"]]
    if model_id in PRETRAINED_MODEL_IDS:
        return [dict(value) for value in modeling_config["pretrained_head_candidates"]]
    raise KeyError(f"unknown nested-validation model: {model_id}")


def rounded_median_epoch(values: Sequence[int]) -> int:
    epochs = np.asarray(values, dtype=np.int64)
    if epochs.ndim != 1 or not len(epochs) or np.any(epochs < 1):
        raise ValueError("best epochs must be a nonempty positive vector")
    return int(math.floor(float(np.median(epochs)) + 0.5))


def summarize_candidate(
    candidate: Mapping[str, Any],
    fold_records: Sequence[Mapping[str, Any]],
    *,
    expected_fold_count: int,
) -> dict[str, Any]:
    if len(fold_records) != int(expected_fold_count):
        raise ValueError("candidate does not cover every required inner fold")
    fold_ids = [str(record["inner_fold_id"]) for record in fold_records]
    if len(fold_ids) != len(set(fold_ids)):
        raise ValueError("candidate contains duplicate inner folds")
    scores = np.asarray(
        [record["macro_average_precision"] for record in fold_records], dtype=np.float64
    )
    if not np.all(np.isfinite(scores)):
        raise ValueError("candidate has nonfinite selection scores")
    parameters = [int(record.get("trainable_parameters") or 0) for record in fold_records]
    latencies = [
        float(record.get("median_encounter_inference_ms") or 0.0)
        for record in fold_records
    ]
    best_epochs = [
        int(record["best_epoch"])
        for record in fold_records
        if record.get("best_epoch") is not None
    ]
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "candidate": dict(candidate),
        "inner_fold_count": len(fold_records),
        "mean_inner_validation_macro_average_precision": float(np.mean(scores)),
        "standard_deviation_inner_validation_macro_average_precision": float(
            np.std(scores)
        ),
        "fold_macro_average_precision": {
            fold_id: float(score)
            for fold_id, score in zip(fold_ids, scores, strict=True)
        },
        "trainable_parameters": max(parameters),
        "mean_median_encounter_inference_ms": float(np.mean(latencies)),
        "inner_best_epochs": best_epochs,
        "selected_final_epoch_if_chosen": (
            rounded_median_epoch(best_epochs) if best_epochs else None
        ),
        "fold_run_ids": [str(record["run_id"]) for record in fold_records],
    }


def select_candidate(
    summaries: Sequence[Mapping[str, Any]],
    *,
    tie_tolerance_macro_ap: float,
) -> dict[str, Any]:
    if not summaries:
        raise ValueError("at least one candidate summary is required")
    tolerance = float(tie_tolerance_macro_ap)
    if tolerance < 0:
        raise ValueError("tie tolerance cannot be negative")
    scores = [
        float(record["mean_inner_validation_macro_average_precision"])
        for record in summaries
    ]
    if not np.all(np.isfinite(scores)):
        raise ValueError("candidate scores must be finite")
    best = max(scores)
    eligible = [
        dict(record)
        for record, score in zip(summaries, scores, strict=True)
        if best - score <= tolerance + 1.0e-15
    ]
    eligible.sort(
        key=lambda record: (
            int(record["trainable_parameters"]),
            float(record["mean_median_encounter_inference_ms"]),
            str(record["candidate_id"]),
        )
    )
    selected = dict(eligible[0])
    selected["selection_best_score_before_tie_break"] = float(best)
    selected["selection_tie_tolerance_macro_ap"] = tolerance
    selected["tie_eligible_candidate_ids"] = [
        str(record["candidate_id"]) for record in eligible
    ]
    selected["tie_break_order_applied"] = [
        "fewer_trainable_parameters",
        "lower_validation_latency",
        "lexicographic_configuration_id",
    ]
    return selected


def assemble_oof_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
    expected_encounter_ids: Sequence[str],
) -> dict[str, Any]:
    expected = tuple(sorted(str(value) for value in expected_encounter_ids))
    if len(expected) != len(set(expected)):
        raise ValueError("expected OOF encounter IDs contain duplicates")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in prediction_rows:
        encounter_id = str(row["encounter_id"])
        if encounter_id in by_id:
            raise ValueError(f"duplicate OOF prediction for {encounter_id}")
        by_id[encounter_id] = row
    if set(by_id) != set(expected):
        raise ValueError("OOF predictions do not exactly cover the outer-training pool")
    ordered = [by_id[value] for value in expected]
    labels = np.asarray([row["y_true"] for row in ordered], dtype=np.int8)
    probabilities = np.asarray([row["probability"] for row in ordered], dtype=np.float64)
    if labels.shape != (len(expected), len(LABEL_ORDER)):
        raise ValueError("OOF labels have an unexpected shape")
    if probabilities.shape != labels.shape or not np.all(np.isfinite(probabilities)):
        raise ValueError("OOF probabilities have an unexpected shape or values")
    return {
        "encounter_ids": expected,
        "campaign_ids": tuple(str(row["campaign_id"]) for row in ordered),
        "labels": labels,
        "probabilities": probabilities,
        "macro_average_precision": macro_average_precision(labels, probabilities),
    }


def _probability_logit(probability: np.ndarray, clip: tuple[float, float]) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), clip[0], clip[1])
    return np.log(clipped / (1.0 - clipped))


def fit_platt_calibrator(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    *,
    c_value: float,
    probability_clip: Sequence[float],
    random_seed: int,
) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=np.int8)
    probability = np.asarray(y_probability, dtype=np.float64)
    if truth.shape != probability.shape or truth.ndim != 2:
        raise ValueError("calibration arrays must be matching matrices")
    clip = (float(probability_clip[0]), float(probability_clip[1]))
    if not 0 < clip[0] < clip[1] < 1 or c_value <= 0:
        raise ValueError("invalid calibration configuration")
    logits = _probability_logit(probability, clip)
    heads = []
    for label_index in range(truth.shape[1]):
        target = truth[:, label_index]
        if len(np.unique(target)) < 2:
            heads.append(
                {
                    "kind": "constant",
                    "probability": float(np.mean(target)),
                }
            )
            continue
        model = LogisticRegression(
            C=float(c_value),
            penalty="l2",
            solver="liblinear",
            random_state=int(random_seed) + label_index,
            max_iter=5000,
        ).fit(logits[:, label_index, None], target)
        heads.append(
            {
                "kind": "platt_logistic",
                "coefficient": float(model.coef_[0, 0]),
                "intercept": float(model.intercept_[0]),
            }
        )
    return {
        "method": "per_label_platt_logistic",
        "c_value": float(c_value),
        "probability_clip": list(clip),
        "random_seed": int(random_seed),
        "label_order": list(LABEL_ORDER),
        "heads": heads,
    }


def identity_calibrator() -> dict[str, Any]:
    return {
        "method": "identity_no_recalibration",
        "label_order": list(LABEL_ORDER),
    }


def apply_calibrator(
    y_probability: np.ndarray,
    calibrator: Mapping[str, Any],
) -> np.ndarray:
    probability = np.asarray(y_probability, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] != len(LABEL_ORDER):
        raise ValueError("calibration input must be an [encounter, 4] matrix")
    if calibrator["method"] == "identity_no_recalibration":
        return probability.copy()
    if calibrator["method"] != "per_label_platt_logistic":
        raise ValueError("unknown calibrator method")
    clip_values = calibrator["probability_clip"]
    logits = _probability_logit(
        probability, (float(clip_values[0]), float(clip_values[1]))
    )
    columns = []
    for label_index, head in enumerate(calibrator["heads"]):
        if head["kind"] == "constant":
            columns.append(
                np.full(len(probability), float(head["probability"]), dtype=np.float64)
            )
        else:
            linear = (
                float(head["coefficient"]) * logits[:, label_index]
                + float(head["intercept"])
            )
            columns.append(1.0 / (1.0 + np.exp(-np.clip(linear, -50.0, 50.0))))
    return np.column_stack(columns)


def select_multilabel_thresholds(
    y_true: np.ndarray,
    y_probability: np.ndarray,
) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=np.int8)
    probability = np.asarray(y_probability, dtype=np.float64)
    if truth.shape != probability.shape or truth.shape[1] != len(LABEL_ORDER):
        raise ValueError("threshold arrays must be matching [encounter, 4] matrices")
    records = {}
    for index, label in enumerate(LABEL_ORDER):
        records[label] = select_largest_max_f1_threshold(
            truth[:, index], probability[:, index]
        )
    return {
        "method": "largest_threshold_among_equal_maximum_inner_oof_f1",
        "label_order": list(LABEL_ORDER),
        "per_label": records,
    }


def ensemble_prediction_matrices(
    matrices: Sequence[np.ndarray],
) -> np.ndarray:
    if not matrices:
        raise ValueError("at least one seed prediction matrix is required")
    values = [np.asarray(matrix, dtype=np.float64) for matrix in matrices]
    shape = values[0].shape
    if any(value.shape != shape for value in values):
        raise ValueError("seed prediction matrices do not share a shape")
    if any(not np.all(np.isfinite(value)) for value in values):
        raise ValueError("seed predictions contain nonfinite values")
    return np.mean(np.stack(values, axis=0), axis=0)
