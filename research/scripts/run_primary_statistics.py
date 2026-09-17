#!/usr/bin/env python3
"""Run the frozen pooled-OOF statistical analysis for the primary cohort."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dolphin_behavior_mil.evaluation import (  # noqa: E402
    holm_adjust,
    paired_campaign_prediction_swap_test,
    percentile_interval,
)
from dolphin_behavior_mil.experiment import (  # noqa: E402
    atomic_write_json,
    sha256_file,
)
from dolphin_behavior_mil.statistical_analysis import (  # noqa: E402
    bootstrap_average_precision,
    decision_metrics,
    draw_valid_campaign_bootstrap_plan,
    probability_metrics,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


NESTED_ROOT = PROJECT_ROOT / "results/nested_cv"
OUTPUT_ROOT = PROJECT_ROOT / "results/statistical_analysis"
OUTER_FOLDS = tuple(f"outer_{index:02d}" for index in range(1, 6))


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError("all CSV rows must have identical ordered fields")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_savez_compressed(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def model_order() -> tuple[str, ...]:
    registry = read_json(
        PROJECT_ROOT / "results/model_implementation/model_registry_v1.json"
    )
    rows = sorted(registry["models"], key=lambda row: int(row["registry_order"]))
    models = tuple(str(row["model_id"]) for row in rows)
    if len(models) != int(registry["required_model_count"]):
        raise RuntimeError("model registry count mismatch")
    return models


def load_aligned_oof(
    models: Sequence[str],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    np.ndarray,
    dict[str, dict[str, np.ndarray]],
    dict[str, list[dict[str, Any]]],
]:
    canonical: tuple[Any, ...] | None = None
    probabilities: dict[str, dict[str, np.ndarray]] = {}
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    for model_id in models:
        metric_path = NESTED_ROOT / "oof" / model_id / "oof_metrics.json"
        prediction_path = NESTED_ROOT / "oof" / model_id / "oof_predictions.jsonl"
        if not metric_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(
                f"official OOF result is incomplete for {model_id}: {prediction_path}"
            )
        if read_json(metric_path).get("status") != "complete":
            raise RuntimeError(f"OOF status is not complete for {model_id}")
        rows = sorted(read_jsonl(prediction_path), key=lambda row: row["encounter_id"])
        encounter_ids = tuple(str(row["encounter_id"]) for row in rows)
        campaign_ids = tuple(str(row["campaign_id"]) for row in rows)
        outer_fold_ids = tuple(str(row["outer_fold_id"]) for row in rows)
        labels = np.asarray([row["y_true"] for row in rows], dtype=np.int8)
        label_orders = {tuple(row["label_order"]) for row in rows}
        if len(rows) != 234 or len(set(encounter_ids)) != 234:
            raise RuntimeError(f"OOF coverage is not exactly 234 encounters: {model_id}")
        if len(label_orders) != 1:
            raise RuntimeError(f"inconsistent label order: {model_id}")
        label_order = next(iter(label_orders))
        signature = (encounter_ids, campaign_ids, outer_fold_ids, label_order, labels)
        if canonical is None:
            canonical = signature
        else:
            for current, expected, name in zip(
                signature[:4],
                canonical[:4],
                ("encounter", "campaign", "outer fold", "label order"),
                strict=True,
            ):
                if current != expected:
                    raise RuntimeError(f"cross-model {name} alignment mismatch: {model_id}")
            if not np.array_equal(labels, canonical[4]):
                raise RuntimeError(f"cross-model label alignment mismatch: {model_id}")
        probabilities[model_id] = {
            "raw": np.asarray([row["raw_probability"] for row in rows], dtype=np.float64),
            "calibrated": np.asarray(
                [row["calibrated_probability"] for row in rows], dtype=np.float64
            ),
        }
        rows_by_model[model_id] = rows
    assert canonical is not None
    return (
        canonical[0],
        canonical[1],
        canonical[2],
        canonical[3],
        canonical[4],
        probabilities,
        rows_by_model,
    )


def optimized_threshold_matrix(
    model_id: str,
    outer_fold_ids: Sequence[str],
    label_order: Sequence[str],
) -> tuple[np.ndarray, dict[str, list[float]]]:
    by_fold: dict[str, list[float]] = {}
    for outer_fold_id in OUTER_FOLDS:
        path = (
            NESTED_ROOT
            / "selection"
            / model_id
            / outer_fold_id
            / "thresholds.json"
        )
        record = read_json(path)
        if tuple(record["label_order"]) != tuple(label_order):
            raise RuntimeError(f"threshold label order mismatch: {model_id}/{outer_fold_id}")
        by_fold[outer_fold_id] = [
            float(record["per_label"][label]["threshold"]) for label in label_order
        ]
    matrix = np.asarray([by_fold[value] for value in outer_fold_ids], dtype=np.float64)
    return matrix, by_fold


def interval(values: np.ndarray, confidence_level: float) -> dict[str, float]:
    lower, upper = percentile_interval(values, confidence_level=confidence_level)
    return {"lower": lower, "upper": upper}


def collect_efficiency(models: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id in models:
        metrics: list[dict[str, Any]] = []
        run_ids: list[str] = []
        for outer_fold_id in OUTER_FOLDS:
            result_path = (
                NESTED_ROOT
                / "outer_folds"
                / model_id
                / outer_fold_id
                / "outer_fold_result.json"
            )
            result = read_json(result_path)
            candidate = str(result["selected_candidate_id"])
            selection_hash = str(result["selection_artifact_sha256"])
            for seed in result["final_seeds"]:
                pattern = (
                    f"outer_final*__{model_id}__{outer_fold_id}__no_inner__"
                    f"{candidate}__{int(seed)}__*"
                )
                candidates = []
                for run_dir in (PROJECT_ROOT / "results/model_runs").glob(pattern):
                    manifest_path = run_dir / "run_manifest.json"
                    metric_path = run_dir / "metrics.json"
                    if not manifest_path.is_file() or not metric_path.is_file():
                        continue
                    manifest = read_json(manifest_path)
                    configuration = read_json(run_dir / "configuration.json")
                    if (
                        manifest.get("status") == "completed"
                        and configuration.get("scope")
                        == "official_outer_test_evaluation_after_locked_inner_selection"
                        and configuration.get("selection_artifact_sha256") == selection_hash
                    ):
                        candidates.append(run_dir)
                if len(candidates) != 1:
                    raise RuntimeError(
                        f"expected one official final run for {model_id}/"
                        f"{outer_fold_id}/{seed}, found {len(candidates)}"
                    )
                run_dir = candidates[0]
                run_ids.append(run_dir.name)
                metrics.append(read_json(run_dir / "metrics.json"))

        def finite_values(field: str) -> list[float]:
            return [
                float(record[field])
                for record in metrics
                if record.get(field) is not None
                and np.isfinite(float(record[field]))
            ]

        parameters = finite_values("trainable_parameters")
        training_seconds = finite_values("training_seconds")
        inference_ms = finite_values("median_encounter_inference_ms")
        rows.append(
            {
                "model_id": model_id,
                "official_final_fit_count": len(metrics),
                "trainable_parameters_median": (
                    float(np.median(parameters)) if parameters else None
                ),
                "trainable_parameters_min": min(parameters) if parameters else None,
                "trainable_parameters_max": max(parameters) if parameters else None,
                "training_seconds_total_observed": (
                    float(np.sum(training_seconds)) if training_seconds else None
                ),
                "training_seconds_median_per_fit": (
                    float(np.median(training_seconds)) if training_seconds else None
                ),
                "training_timing_available_fit_count": len(training_seconds),
                "median_encounter_inference_ms_across_fits": (
                    float(np.median(inference_ms)) if inference_ms else None
                ),
                "inference_timing_available_fit_count": len(inference_ms),
                "peak_memory_mb": None,
                "peak_memory_undefined_reason": (
                    "the frozen runner did not instrument process-level peak memory"
                ),
                "parameter_scope_note": (
                    "trainable classifier/head only for frozen-embedding models; "
                    "pretrained encoder extraction cost is excluded"
                    if model_id in {"panns_frozen_mil", "aves_frozen_mil"}
                    else "all trainable parameters in the fitted model"
                ),
                "run_ids": run_ids,
            }
        )
    return rows


def main() -> int:
    validation_path = PROJECT_ROOT / "configs/validation_protocol_v1.toml"
    validation = read_toml(validation_path)
    models = model_order()
    (
        encounter_ids,
        campaign_ids,
        outer_fold_ids,
        label_order,
        truth,
        probabilities,
        _,
    ) = load_aligned_oof(models)
    uncertainty = validation["uncertainty"]
    paired = validation["paired_tests"]
    confidence_level = float(uncertainty["confidence_level"])

    metrics: dict[str, Any] = {}
    optimized_thresholds: dict[str, dict[str, list[float]]] = {}
    for model_id in models:
        raw = probabilities[model_id]["raw"]
        calibrated = probabilities[model_id]["calibrated"]
        threshold_matrix, by_fold = optimized_threshold_matrix(
            model_id, outer_fold_ids, label_order
        )
        optimized_thresholds[model_id] = by_fold
        metrics[model_id] = {
            "raw_probability": probability_metrics(
                truth,
                raw,
                label_order,
                roc_auc_minimum_positive=int(
                    validation["metrics"]["roc_auc_minimum_positive"]
                ),
                roc_auc_minimum_negative=int(
                    validation["metrics"]["roc_auc_minimum_negative"]
                ),
            ),
            "calibrated_probability": probability_metrics(
                truth,
                calibrated,
                label_order,
                roc_auc_minimum_positive=int(
                    validation["metrics"]["roc_auc_minimum_positive"]
                ),
                roc_auc_minimum_negative=int(
                    validation["metrics"]["roc_auc_minimum_negative"]
                ),
            ),
            "raw_fixed_0_5_decision": decision_metrics(
                truth, raw, 0.5, label_order
            ),
            "calibrated_fixed_0_5_decision": decision_metrics(
                truth, calibrated, 0.5, label_order
            ),
            "calibrated_inner_optimized_decision": decision_metrics(
                truth, calibrated, threshold_matrix, label_order
            ),
        }

    print(
        f"drawing {int(uncertainty['resamples'])} valid campaign bootstrap replicates",
        flush=True,
    )
    plan = draw_valid_campaign_bootstrap_plan(
        truth,
        campaign_ids,
        resamples=int(uncertainty["resamples"]),
        seed=int(uncertainty["seed"]),
    )
    replicate_arrays: dict[str, np.ndarray] = {
        "campaign_draws": plan.draws,
        "campaign_order": np.asarray(plan.campaign_order, dtype=str),
    }
    bootstrap: dict[str, Any] = {}
    macro_replicates: dict[str, dict[str, np.ndarray]] = {}
    for model_id in models:
        print(f"bootstrap AP: {model_id}", flush=True)
        bootstrap[model_id] = {}
        macro_replicates[model_id] = {}
        for probability_kind in ("raw", "calibrated"):
            macro, per_label = bootstrap_average_precision(
                truth, probabilities[model_id][probability_kind], plan
            )
            macro_replicates[model_id][probability_kind] = macro
            replicate_arrays[f"{probability_kind}__{model_id}__macro_ap"] = macro
            replicate_arrays[f"{probability_kind}__{model_id}__per_label_ap"] = per_label
            bootstrap[model_id][probability_kind] = {
                "macro_average_precision": {
                    "point_estimate": metrics[model_id][
                        f"{probability_kind}_probability"
                    ]["macro_average_precision"],
                    **interval(macro, confidence_level),
                },
                "per_label_average_precision": {
                    label: {
                        "point_estimate": metrics[model_id][
                            f"{probability_kind}_probability"
                        ]["per_label_average_precision"][label],
                        **interval(per_label[:, index], confidence_level),
                    }
                    for index, label in enumerate(label_order)
                },
            }

    confirmatory: list[dict[str, Any]] = []
    raw_p_values: dict[str, float] = {}
    for contrast in validation["confirmatory_contrasts"]:
        contrast_id = str(contrast["contrast_id"])
        model_a = str(contrast["model_a"])
        model_b = str(contrast["model_b"])
        test = paired_campaign_prediction_swap_test(
            truth,
            probabilities[model_a]["raw"],
            probabilities[model_b]["raw"],
            campaign_ids,
            permutations=int(paired["permutations"]),
            seed=int(paired["seed"]),
        )
        delta_replicates = (
            macro_replicates[model_a]["raw"]
            - macro_replicates[model_b]["raw"]
        )
        delta_interval = interval(delta_replicates, confidence_level)
        record = {
            "contrast_id": contrast_id,
            "model_a": model_a,
            "model_b": model_b,
            "metric": str(contrast["metric"]),
            "probability_kind": "raw",
            "observed_delta_a_minus_b": test.observed_delta,
            "bootstrap_confidence_interval": delta_interval,
            "unadjusted_p_value": test.p_value,
            "permutations": test.permutations,
            "exceedance_count": test.exceedance_count,
        }
        confirmatory.append(record)
        raw_p_values[contrast_id] = test.p_value
    adjusted = holm_adjust(raw_p_values)
    alpha = float(paired["alpha"])
    for record in confirmatory:
        record["holm_adjusted_p_value"] = adjusted[record["contrast_id"]]
        record["reject_at_familywise_alpha_0_05"] = bool(
            adjusted[record["contrast_id"]] <= alpha
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_ROOT / "primary_metrics_v1.json"
    bootstrap_path = OUTPUT_ROOT / "bootstrap_intervals_v1.json"
    replicate_path = OUTPUT_ROOT / "bootstrap_ap_replicates_v1.npz"
    confirmatory_path = OUTPUT_ROOT / "confirmatory_tests_v1.json"
    thresholds_path = OUTPUT_ROOT / "optimized_thresholds_by_outer_fold_v1.json"
    efficiency_path = OUTPUT_ROOT / "efficiency_metrics_v1.json"
    atomic_write_json(
        metrics_path,
        {
            "analysis_unit": "encounter",
            "encounter_count": len(encounter_ids),
            "campaign_count": len(set(campaign_ids)),
            "label_order": list(label_order),
            "model_order": list(models),
            "primary_probability_kind": "raw",
            "metrics": metrics,
        },
    )
    atomic_write_json(
        bootstrap_path,
        {
            "method": str(uncertainty["method"]),
            "resamples": plan.resamples,
            "seed": plan.seed,
            "attempts": plan.attempts,
            "confidence_level": confidence_level,
            "campaign_count": len(plan.campaign_order),
            "intervals": bootstrap,
        },
    )
    atomic_savez_compressed(replicate_path, **replicate_arrays)
    atomic_write_json(
        confirmatory_path,
        {
            "method": str(paired["method"]),
            "alternative": str(paired["alternative"]),
            "multiple_testing": str(paired["multiple_testing"]),
            "familywise_alpha": alpha,
            "contrasts": confirmatory,
        },
    )
    atomic_write_json(thresholds_path, optimized_thresholds)
    efficiency = collect_efficiency(models)
    atomic_write_json(
        efficiency_path,
        {
            "model_order": list(models),
            "models": efficiency,
            "timing_interpretation": (
                "descriptive measurements from the recorded local runs; hardware and "
                "concurrent workload were not standardized for a formal speed test"
            ),
        },
    )

    primary_rows = []
    per_label_rows = []
    for model_id in models:
        raw_metrics = metrics[model_id]["raw_probability"]
        calibrated_metrics = metrics[model_id]["calibrated_probability"]
        raw_interval = bootstrap[model_id]["raw"]["macro_average_precision"]
        calibrated_interval = bootstrap[model_id]["calibrated"][
            "macro_average_precision"
        ]
        optimized = metrics[model_id]["calibrated_inner_optimized_decision"]
        fixed = metrics[model_id]["calibrated_fixed_0_5_decision"]
        primary_rows.append(
            {
                "model_id": model_id,
                "raw_macro_ap": raw_metrics["macro_average_precision"],
                "raw_macro_ap_ci_lower": raw_interval["lower"],
                "raw_macro_ap_ci_upper": raw_interval["upper"],
                "raw_macro_roc_auc": raw_metrics["macro_roc_auc"],
                "calibrated_macro_ap": calibrated_metrics["macro_average_precision"],
                "calibrated_macro_ap_ci_lower": calibrated_interval["lower"],
                "calibrated_macro_ap_ci_upper": calibrated_interval["upper"],
                "calibrated_macro_brier": calibrated_metrics["macro_brier_score"],
                "calibrated_macro_ece": calibrated_metrics["macro_adaptive_ece"],
                "calibrated_macro_f1_fixed_0_5": fixed["macro_f1"],
                "calibrated_macro_f1_inner_optimized": optimized["macro_f1"],
                "calibrated_hamming_loss_inner_optimized": optimized["hamming_loss"],
            }
        )
        for label in label_order:
            raw_ci = bootstrap[model_id]["raw"]["per_label_average_precision"][label]
            calibrated_ci = bootstrap[model_id]["calibrated"][
                "per_label_average_precision"
            ][label]
            per_label_rows.append(
                {
                    "model_id": model_id,
                    "label": label,
                    "positive_count": raw_metrics["positive_counts"][label],
                    "negative_count": raw_metrics["negative_counts"][label],
                    "raw_ap": raw_metrics["per_label_average_precision"][label],
                    "raw_ap_ci_lower": raw_ci["lower"],
                    "raw_ap_ci_upper": raw_ci["upper"],
                    "raw_roc_auc": raw_metrics["per_label_roc_auc"][label]["value"],
                    "calibrated_ap": calibrated_metrics[
                        "per_label_average_precision"
                    ][label],
                    "calibrated_ap_ci_lower": calibrated_ci["lower"],
                    "calibrated_ap_ci_upper": calibrated_ci["upper"],
                    "calibrated_brier": calibrated_metrics[
                        "per_label_brier_score"
                    ][label],
                    "calibrated_ece": calibrated_metrics[
                        "per_label_adaptive_ece"
                    ][label],
                    "calibrated_f1_inner_optimized": optimized["per_label_f1"][label],
                }
            )
    primary_csv = OUTPUT_ROOT / "primary_model_table_v1.csv"
    per_label_csv = OUTPUT_ROOT / "per_label_table_v1.csv"
    atomic_write_csv(primary_csv, primary_rows)
    atomic_write_csv(per_label_csv, per_label_rows)

    primary_scores = {
        model_id: metrics[model_id]["raw_probability"]["macro_average_precision"]
        for model_id in models
    }
    ranking = sorted(primary_scores, key=lambda key: (-primary_scores[key], key))
    source_paths = [
        validation_path,
        PROJECT_ROOT / "results/model_implementation/model_registry_v1.json",
    ] + [
        NESTED_ROOT / "oof" / model_id / "oof_predictions.jsonl"
        for model_id in models
    ] + [
        NESTED_ROOT
        / "selection"
        / model_id
        / outer_fold_id
        / "thresholds.json"
        for model_id in models
        for outer_fold_id in OUTER_FOLDS
    ] + [
        NESTED_ROOT
        / "outer_folds"
        / model_id
        / outer_fold_id
        / "outer_fold_result.json"
        for model_id in models
        for outer_fold_id in OUTER_FOLDS
    ] + [
        PROJECT_ROOT / "results/model_runs" / run_id / filename
        for model_record in efficiency
        for run_id in model_record["run_ids"]
        for filename in ("run_manifest.json", "configuration.json", "metrics.json")
    ]
    output_paths = [
        metrics_path,
        bootstrap_path,
        replicate_path,
        confirmatory_path,
        thresholds_path,
        efficiency_path,
        primary_csv,
        per_label_csv,
    ]
    summary_path = OUTPUT_ROOT / "step_10_primary_summary.json"
    summary = {
        "status": "complete",
        "scope": "prespecified_primary_pooled_oof_analysis",
        "encounter_count": len(encounter_ids),
        "campaign_count": len(set(campaign_ids)),
        "model_count": len(models),
        "label_order": list(label_order),
        "best_raw_macro_ap_model": ranking[0],
        "raw_macro_ap_ranking": [
            {"rank": index, "model_id": model_id, "raw_macro_ap": primary_scores[model_id]}
            for index, model_id in enumerate(ranking, start=1)
        ],
        "hf_lw_gam_rank": ranking.index("hf_lw_gam") + 1,
        "confirmatory_rejections_after_holm": [
            record["contrast_id"]
            for record in confirmatory
            if record["reject_at_familywise_alpha_0_05"]
        ],
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in source_paths
        },
        "artifact_sha256": {
            str(path.relative_to(OUTPUT_ROOT)): sha256_file(path)
            for path in output_paths
        },
        "evidence_boundary": (
            "all labels and claims remain encounter-level; no result supplies a "
            "per-whistle behavior ground truth or validates at-sea real-time deployment"
        ),
    }
    atomic_write_json(summary_path, summary)
    print(
        f"complete: best raw macro-AP={ranking[0]} ({primary_scores[ranking[0]]:.6f}); "
        f"HF-LW-GAM rank={summary['hf_lw_gam_rank']}/{len(models)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
