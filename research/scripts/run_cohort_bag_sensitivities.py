#!/usr/bin/env python3
"""Run frozen OOF subset and locked-HF cohort/bag sensitivity analyses."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dolphin_behavior_mil.evaluation import percentile_interval  # noqa: E402
from dolphin_behavior_mil.experiment import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    load_toml,
    sha256_file,
)
from dolphin_behavior_mil.mil_models import build_cnn_mil_model  # noqa: E402
from dolphin_behavior_mil.modeling_data import (  # noqa: E402
    DatasetRegistry,
    FeatureNormalizer,
    LABEL_ORDER,
    MILBagDataset,
    NormalizedFeatureCache,
    read_jsonl,
)
from dolphin_behavior_mil.nested_validation import apply_calibrator  # noqa: E402
from dolphin_behavior_mil.statistical_analysis import (  # noqa: E402
    bootstrap_average_precision,
    draw_valid_campaign_bootstrap_plan,
    probability_metrics,
)
from dolphin_behavior_mil.training import predict_mil  # noqa: E402


MODEL_ID = "hf_lw_gam"
OUTER_FOLDS = tuple(f"outer_{index:02d}" for index in range(1, 6))
MODEL_ORDER = (
    "prior_constant",
    "bag_size_logistic",
    "metadata_logistic",
    "handcrafted_logistic",
    "cnn_mean",
    "cnn_max",
    "cnn_linear_softmax",
    "cnn_shared_gated_attention",
    "hf_lw_gam",
    "panns_frozen_mil",
    "aves_frozen_mil",
)
SUBSET_COHORTS = (
    ("primary_234", "primary_234"),
    ("scans_ge_2_218", "sensitivity_scans_ge_2_218"),
    ("behavior_sum_equal_scans_121", "sensitivity_behavior_sum_equal_121"),
)
BAG_FIELDS = {
    "all_segments_4152": "all_segments_sensitivity_instance",
    "high_confidence_3884": "high_confidence_sensitivity_instance",
}
RESULT_ROOT = PROJECT_ROOT / "results/sensitivity/cohort_bag"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def official_run_directory(
    outer_fold_id: str,
    candidate_id: str,
    seed: int,
    selection_sha256: str,
) -> Path:
    pattern = (
        f"outer_final_cnn_compute_v1__{MODEL_ID}__{outer_fold_id}__no_inner__"
        f"{candidate_id}__{int(seed)}__*"
    )
    matches: list[Path] = []
    for path in (PROJECT_ROOT / "results/model_runs").glob(pattern):
        manifest_path = path / "run_manifest.json"
        configuration_path = path / "configuration.json"
        if not manifest_path.is_file() or not configuration_path.is_file():
            continue
        manifest = read_json(manifest_path)
        configuration = read_json(configuration_path)
        if (
            manifest.get("status") == "completed"
            and configuration.get("scope")
            == "official_outer_test_evaluation_after_locked_inner_selection"
            and configuration.get("selection_artifact_sha256") == selection_sha256
        ):
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one official checkpoint for {outer_fold_id}/{seed}; "
            f"found {len(matches)}"
        )
    return matches[0]


def build_model(candidate: Mapping[str, Any], encoder_microbatch: int) -> Any:
    return build_cnn_mil_model(
        MODEL_ID,
        base_channels=int(candidate["base_channels"]),
        embedding_dim=int(candidate["embedding_dim"]),
        attention_dim=int(candidate["attention_dim"]),
        dropout=float(candidate["dropout"]),
        encoder_microbatch_max_instances=int(encoder_microbatch),
    )


def clone_registry(
    base: DatasetRegistry,
    *,
    whistles: Mapping[str, tuple[Mapping[str, Any], ...]],
    labels: Mapping[str, np.ndarray] | None = None,
    encounters: Mapping[str, Mapping[str, Any]] | None = None,
    environments: Mapping[str, Mapping[str, Any]] | None = None,
    campaign_by_encounter: Mapping[str, str] | None = None,
    outer_fold_by_encounter: Mapping[str, str] | None = None,
) -> DatasetRegistry:
    return DatasetRegistry(
        project_root=base.project_root,
        version_dir=base.version_dir,
        encounters=base.encounters if encounters is None else encounters,
        labels=base.labels if labels is None else labels,
        whistles=whistles,
        environments=base.environments if environments is None else environments,
        campaign_by_encounter=(
            base.campaign_by_encounter
            if campaign_by_encounter is None
            else campaign_by_encounter
        ),
        outer_fold_by_encounter=(
            base.outer_fold_by_encounter
            if outer_fold_by_encounter is None
            else outer_fold_by_encounter
        ),
        inner_rows=base.inner_rows,
        feature_values_by_whistle=base.feature_values_by_whistle,
    )


def ordered_whistles(
    rows: Sequence[Mapping[str, Any]],
    encounter_ids: Sequence[str],
    *,
    flag: str | None = None,
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    expected = set(encounter_ids)
    grouped: dict[str, list[Mapping[str, Any]]] = {value: [] for value in expected}
    for row in rows:
        encounter_id = str(row["encounter_id"])
        if encounter_id not in expected:
            continue
        if flag is not None and not bool(row.get(flag)):
            continue
        if row.get("feature_cache") is None:
            raise RuntimeError(f"missing cached feature: {row['whistle_id']}")
        grouped[encounter_id].append(row)
    result: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for encounter_id, values in grouped.items():
        values.sort(key=lambda row: (int(row["sequence_index"]), str(row["whistle_id"])))
        sequence = [int(row["sequence_index"]) for row in values]
        if not values or len(sequence) != len(set(sequence)):
            raise RuntimeError(f"invalid sensitivity bag: {encounter_id}")
        result[encounter_id] = tuple(values)
    return result


def interval(values: np.ndarray) -> dict[str, float]:
    lower, upper = percentile_interval(values, confidence_level=0.95)
    return {"lower": float(lower), "upper": float(upper)}


def metric_and_interval(
    truth: np.ndarray,
    probability: np.ndarray,
    campaigns: Sequence[str],
    *,
    resamples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = probability_metrics(truth, probability, LABEL_ORDER)
    plan = draw_valid_campaign_bootstrap_plan(
        truth,
        campaigns,
        resamples=resamples,
        seed=seed,
    )
    macro, per_label = bootstrap_average_precision(truth, probability, plan)
    intervals = {
        "resamples": plan.resamples,
        "campaign_count": len(plan.campaign_order),
        "macro_average_precision": {
            "point_estimate": metrics["macro_average_precision"],
            **interval(macro),
        },
        "per_label_average_precision": {
            label: {
                "point_estimate": metrics["per_label_average_precision"][label],
                **interval(per_label[:, index]),
            }
            for index, label in enumerate(LABEL_ORDER)
        },
    }
    return metrics, intervals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(int(args.torch_threads))
    device = torch.device(str(args.device))
    execution_path = PROJECT_ROOT / "configs/cohort_bag_sensitivity_v1.toml"
    execution = load_toml(execution_path)
    private_path = PROJECT_ROOT / "configs/cohort_bag_private.toml"
    if not private_path.is_file():
        raise FileNotFoundError(
            "Numeric-cohort identifiers are confidential. Supply the local "
            "configs/cohort_bag_private.toml described in research/README.md."
        )
    private_settings = load_toml(private_path)
    execution["numeric_behavior_236"].update(private_settings)
    numeric_extra_ids = tuple(private_settings["additional_encounters"])
    expected_extra_instances = int(private_settings["expected_extra_instance_count"])
    if len(numeric_extra_ids) != 2 or expected_extra_instances <= 0:
        raise ValueError("Expected two local numeric-cohort identifiers and a positive clip count")
    if not bool(execution["frozen_before_any_cohort_or_alternative_bag_result"]):
        raise RuntimeError("sensitivity execution was not frozen")
    modeling = load_toml(PROJECT_ROOT / "configs/modeling_protocol_v1.toml")
    validation = load_toml(PROJECT_ROOT / "configs/validation_protocol_v1.toml")
    base = DatasetRegistry.load(PROJECT_ROOT, cache_features_in_memory=True)
    base.preload_feature_values()
    all_whistle_rows = read_jsonl(base.version_dir / "whistles.jsonl")
    cohort_rows = read_jsonl(base.version_dir / "cohorts.jsonl")
    cohort_by_id = {str(row["encounter_id"]): row for row in cohort_rows}

    alternative_registries: dict[str, DatasetRegistry] = {}
    expected_bag_counts = {"all_segments_4152": 4152, "high_confidence_3884": 3884}
    for variant, flag in BAG_FIELDS.items():
        whistles = ordered_whistles(all_whistle_rows, base.encounter_ids, flag=flag)
        if sum(len(values) for values in whistles.values()) != expected_bag_counts[variant]:
            raise RuntimeError(f"alternative bag count mismatch: {variant}")
        alternative_registries[variant] = clone_registry(base, whistles=whistles)

    encounter_rows = read_jsonl(base.version_dir / "encounters.jsonl")
    encounters_all = {str(row["encounter_id"]): row for row in encounter_rows}
    environment_rows = read_jsonl(base.version_dir / "environment_flat.jsonl")
    environment_by_source = {
        int(row["source_csv_row_number"]): row for row in environment_rows
    }
    numeric_label_rows = read_jsonl(
        base.version_dir / "labels_sensitivity_numeric.jsonl"
    )
    numeric_label_by_id = {
        str(row["encounter_id"]): row for row in numeric_label_rows
    }
    numeric_labels = dict(base.labels)
    numeric_encounters = dict(base.encounters)
    numeric_environments = dict(base.environments)
    numeric_campaigns = dict(base.campaign_by_encounter)
    numeric_outer = dict(base.outer_fold_by_encounter)
    for encounter_id in numeric_extra_ids:
        row = numeric_label_by_id[encounter_id]
        numeric_labels[encounter_id] = np.asarray(
            row["primary_label_vector"], dtype=np.float32
        )
        numeric_encounters[encounter_id] = encounters_all[encounter_id]
        source_row = int(encounters_all[encounter_id]["selected_environment_row"])
        numeric_environments[encounter_id] = environment_by_source[source_row]
        numeric_campaigns[encounter_id] = str(
            execution["numeric_behavior_236"]["additional_encounter_campaign"]
        )
        numeric_outer[encounter_id] = str(
            execution["numeric_behavior_236"]["additional_encounter_outer_fold"]
        )
    numeric_whistles = dict(base.whistles)
    numeric_whistles.update(
        ordered_whistles(all_whistle_rows, numeric_extra_ids, flag=None)
    )
    if sum(len(numeric_whistles[value]) for value in numeric_extra_ids) != expected_extra_instances:
        raise RuntimeError("numeric sensitivity clip count differs from the private configuration")
    numeric_registry = clone_registry(
        base,
        whistles=numeric_whistles,
        labels=numeric_labels,
        encounters=numeric_encounters,
        environments=numeric_environments,
        campaign_by_encounter=numeric_campaigns,
        outer_fold_by_encounter=numeric_outer,
    )

    official_rows = sorted(
        read_jsonl(
            PROJECT_ROOT
            / "results/nested_cv/oof/hf_lw_gam/oof_predictions.jsonl"
        ),
        key=lambda row: row["encounter_id"],
    )
    official_by_id = {str(row["encounter_id"]): row for row in official_rows}
    predicted_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    numeric_extra_predictions: list[dict[str, Any]] = []
    source_paths: list[Path] = [
        execution_path,
        PROJECT_ROOT / "configs/modeling_protocol_v1.toml",
        PROJECT_ROOT / "configs/validation_protocol_v1.toml",
        PROJECT_ROOT / "scripts/run_cohort_bag_sensitivities.py",
        base.version_dir / "cohorts.jsonl",
        base.version_dir / "preprocessing_cohorts.jsonl",
        base.version_dir / "labels_sensitivity_numeric.jsonl",
        base.version_dir / "whistles.jsonl",
    ]
    full_reproduction_difference = 0.0
    for outer_fold_id in OUTER_FOLDS:
        selection_path = (
            PROJECT_ROOT
            / "results/nested_cv/selection/hf_lw_gam"
            / outer_fold_id
            / "candidate_selection.json"
        )
        result_path = (
            PROJECT_ROOT
            / "results/nested_cv/outer_folds/hf_lw_gam"
            / outer_fold_id
            / "outer_fold_result.json"
        )
        selection = read_json(selection_path)
        result = read_json(result_path)
        candidate = dict(selection["selected_candidate"]["candidate"])
        candidate_id = str(candidate["candidate_id"])
        selection_sha = sha256_file(selection_path)
        if candidate_id != result["selected_candidate_id"] or selection_sha != result[
            "selection_artifact_sha256"
        ]:
            raise RuntimeError(f"official selection mismatch: {outer_fold_id}")
        _, test_ids = base.outer_split(outer_fold_id)
        run_dirs = [
            official_run_directory(
                outer_fold_id, candidate_id, int(seed), selection_sha
            )
            for seed in result["final_seeds"]
        ]
        normalizer_records = [read_json(path / "normalizer.json") for path in run_dirs]
        if any(value != normalizer_records[0] for value in normalizer_records[1:]):
            raise RuntimeError(f"seed normalizers differ: {outer_fold_id}")
        normalizer = FeatureNormalizer.from_dict(normalizer_records[0])
        registries: dict[str, DatasetRegistry] = {
            "main_discrete_4126": base,
            **alternative_registries,
        }
        datasets: dict[str, MILBagDataset] = {}
        for variant, registry in registries.items():
            normalized = NormalizedFeatureCache.build(registry, normalizer, test_ids)
            datasets[variant] = MILBagDataset(
                registry,
                test_ids,
                normalizer,
                training=False,
                normalized_feature_cache=normalized,
            )
        numeric_ids: tuple[str, ...] = ()
        if outer_fold_id == str(
            execution["numeric_behavior_236"]["additional_encounter_outer_fold"]
        ):
            numeric_ids = tuple(sorted(numeric_extra_ids))
            numeric_normalized = NormalizedFeatureCache.build(
                numeric_registry, normalizer, numeric_ids
            )
            datasets["numeric_extra_2"] = MILBagDataset(
                numeric_registry,
                numeric_ids,
                normalizer,
                training=False,
                normalized_feature_cache=numeric_normalized,
            )
        matrices: dict[str, list[np.ndarray]] = defaultdict(list)
        references: dict[str, Any] = {}
        for run_dir in run_dirs:
            checkpoint = torch.load(
                run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=True
            )
            model = build_model(
                candidate,
                int(modeling["neural_training"]["encoder_microbatch_max_instances"]),
            )
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.to(device)
            for variant, dataset in datasets.items():
                prediction = predict_mil(
                    model,
                    dataset,
                    batch_size=int(candidate["encounter_batch_size"]),
                    device=device,
                    data_loader_workers=0,
                )
                matrices[variant].append(prediction.probabilities)
                references[variant] = prediction
            del model
        calibrator_path = selection_path.parent / "calibrator.json"
        calibrator = read_json(calibrator_path)
        for variant in registries:
            prediction = references[variant]
            raw = np.mean(np.stack(matrices[variant]), axis=0)
            calibrated = apply_calibrator(raw, calibrator)
            if variant == "main_discrete_4126":
                official = np.asarray(
                    [official_by_id[value]["raw_probability"] for value in prediction.encounter_ids],
                    dtype=np.float64,
                )
                full_reproduction_difference = max(
                    full_reproduction_difference,
                    float(np.max(np.abs(raw - official))),
                )
            for index, encounter_id in enumerate(prediction.encounter_ids):
                predicted_by_variant[variant].append(
                    {
                        "analysis_id": variant,
                        "model_id": MODEL_ID,
                        "outer_fold_id": outer_fold_id,
                        "encounter_id": encounter_id,
                        "campaign_id": prediction.campaign_ids[index],
                        "instance_count": len(registries[variant].whistles[encounter_id]),
                        "label_order": list(LABEL_ORDER),
                        "y_true": prediction.labels[index].astype(int).tolist(),
                        "seed_probabilities": [
                            matrix[index].astype(float).tolist()
                            for matrix in matrices[variant]
                        ],
                        "raw_probability": raw[index].astype(float).tolist(),
                        "calibrated_probability": calibrated[index].astype(float).tolist(),
                        "label_scope": "encounter_bag",
                    }
                )
        if numeric_ids:
            prediction = references["numeric_extra_2"]
            raw = np.mean(np.stack(matrices["numeric_extra_2"]), axis=0)
            calibrated = apply_calibrator(raw, calibrator)
            for index, encounter_id in enumerate(prediction.encounter_ids):
                numeric_extra_predictions.append(
                    {
                        "analysis_id": "numeric_behavior_236",
                        "model_id": MODEL_ID,
                        "outer_fold_id": outer_fold_id,
                        "encounter_id": encounter_id,
                        "campaign_id": prediction.campaign_ids[index],
                        "instance_count": len(numeric_registry.whistles[encounter_id]),
                        "label_order": list(LABEL_ORDER),
                        "y_true": prediction.labels[index].astype(int).tolist(),
                        "seed_probabilities": [
                            matrix[index].astype(float).tolist()
                            for matrix in matrices["numeric_extra_2"]
                        ],
                        "raw_probability": raw[index].astype(float).tolist(),
                        "calibrated_probability": calibrated[index].astype(float).tolist(),
                        "label_scope": "encounter_bag",
                    }
                )
        source_paths.extend(
            [selection_path, result_path, calibrator_path]
            + [
                run_dir / name
                for run_dir in run_dirs
                for name in (
                    "run_manifest.json",
                    "configuration.json",
                    "normalizer.json",
                    "best_checkpoint.pt",
                )
            ]
        )
        print(f"locked sensitivity inference complete: {outer_fold_id}", flush=True)
    if full_reproduction_difference > 1.0e-7:
        raise RuntimeError(
            "locked main-bag inference failed official OOF reproduction: "
            f"{full_reproduction_difference}"
        )

    uncertainty = validation["uncertainty"]
    resamples = int(uncertainty["resamples"])
    seed = int(uncertainty["seed"])
    subset_metrics: dict[str, dict[str, Any]] = {}
    subset_intervals: dict[str, dict[str, Any]] = {}
    for cohort_id, field in SUBSET_COHORTS:
        ids = tuple(
            sorted(
                encounter_id
                for encounter_id in base.encounter_ids
                if bool(cohort_by_id[encounter_id][field])
            )
        )
        subset_metrics[cohort_id] = {}
        subset_intervals[cohort_id] = {}
        for model_id in MODEL_ORDER:
            rows = {
                str(row["encounter_id"]): row
                for row in read_jsonl(
                    PROJECT_ROOT
                    / "results/nested_cv/oof"
                    / model_id
                    / "oof_predictions.jsonl"
                )
            }
            truth = np.asarray([rows[value]["y_true"] for value in ids], dtype=np.int8)
            raw = np.asarray(
                [rows[value]["raw_probability"] for value in ids], dtype=np.float64
            )
            calibrated = np.asarray(
                [rows[value]["calibrated_probability"] for value in ids],
                dtype=np.float64,
            )
            campaigns = tuple(str(rows[value]["campaign_id"]) for value in ids)
            raw_metrics, raw_intervals = metric_and_interval(
                truth, raw, campaigns, resamples=resamples, seed=seed
            )
            subset_metrics[cohort_id][model_id] = {
                "raw": raw_metrics,
                "calibrated": probability_metrics(
                    truth, calibrated, LABEL_ORDER
                ),
            }
            subset_intervals[cohort_id][model_id] = raw_intervals
        print(f"OOF subset complete: {cohort_id}", flush=True)

    locked_metrics: dict[str, Any] = {"cohorts": {}, "bag_variants": {}}
    locked_intervals: dict[str, Any] = {"cohorts": {}, "bag_variants": {}}
    main_rows = {
        str(row["encounter_id"]): row
        for row in predicted_by_variant["main_discrete_4126"]
    }
    for cohort_id, field in SUBSET_COHORTS:
        ids = tuple(
            sorted(
                encounter_id
                for encounter_id in base.encounter_ids
                if bool(cohort_by_id[encounter_id][field])
            )
        )
        truth = np.asarray([main_rows[value]["y_true"] for value in ids], dtype=np.int8)
        raw = np.asarray(
            [main_rows[value]["raw_probability"] for value in ids], dtype=np.float64
        )
        campaigns = tuple(str(main_rows[value]["campaign_id"]) for value in ids)
        metrics, intervals = metric_and_interval(
            truth, raw, campaigns, resamples=resamples, seed=seed
        )
        locked_metrics["cohorts"][cohort_id] = metrics
        locked_intervals["cohorts"][cohort_id] = intervals

    numeric_rows = [
        {
            "analysis_id": "numeric_behavior_236",
            "model_id": MODEL_ID,
            "outer_fold_id": row["outer_fold_id"],
            "encounter_id": row["encounter_id"],
            "campaign_id": row["campaign_id"],
            "instance_count": len(base.whistles[str(row["encounter_id"])]),
            "label_order": list(LABEL_ORDER),
            "y_true": row["y_true"],
            "seed_probabilities": row["seed_probabilities"],
            "raw_probability": row["raw_probability"],
            "calibrated_probability": row["calibrated_probability"],
            "label_scope": "encounter_bag",
        }
        for row in official_rows
    ] + numeric_extra_predictions
    numeric_rows.sort(key=lambda row: row["encounter_id"])
    if len(numeric_rows) != 236 or len({row["encounter_id"] for row in numeric_rows}) != 236:
        raise RuntimeError("numeric sensitivity coverage is not 236")
    numeric_truth = np.asarray([row["y_true"] for row in numeric_rows], dtype=np.int8)
    numeric_raw = np.asarray(
        [row["raw_probability"] for row in numeric_rows], dtype=np.float64
    )
    numeric_campaigns_values = tuple(str(row["campaign_id"]) for row in numeric_rows)
    metrics, intervals = metric_and_interval(
        numeric_truth,
        numeric_raw,
        numeric_campaigns_values,
        resamples=resamples,
        seed=seed,
    )
    locked_metrics["cohorts"]["numeric_behavior_236"] = metrics
    locked_intervals["cohorts"]["numeric_behavior_236"] = intervals

    for variant, rows in predicted_by_variant.items():
        values = sorted(rows, key=lambda row: row["encounter_id"])
        if len(values) != 234:
            raise RuntimeError(f"bag sensitivity coverage mismatch: {variant}")
        truth = np.asarray([row["y_true"] for row in values], dtype=np.int8)
        raw = np.asarray(
            [row["raw_probability"] for row in values], dtype=np.float64
        )
        campaigns = tuple(str(row["campaign_id"]) for row in values)
        metrics, intervals = metric_and_interval(
            truth, raw, campaigns, resamples=resamples, seed=seed
        )
        locked_metrics["bag_variants"][variant] = metrics
        locked_intervals["bag_variants"][variant] = intervals

    prediction_rows = [
        row
        for variant in (
            "main_discrete_4126",
            "all_segments_4152",
            "high_confidence_3884",
        )
        for row in sorted(
            predicted_by_variant[variant], key=lambda value: value["encounter_id"]
        )
    ] + numeric_rows
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    prediction_path = RESULT_ROOT / "locked_hf_predictions_v1.jsonl"
    metric_path = RESULT_ROOT / "cohort_bag_metrics_v1.json"
    interval_path = RESULT_ROOT / "cohort_bag_bootstrap_intervals_v1.json"
    atomic_write_jsonl(prediction_path, prediction_rows)
    atomic_write_json(
        metric_path,
        {
            "status": "complete",
            "label_order": list(LABEL_ORDER),
            "subset_oof_all_models": subset_metrics,
            "locked_hf_lw_gam": locked_metrics,
            "full_bag_max_absolute_difference_from_official_oof": full_reproduction_difference,
        },
    )
    atomic_write_json(
        interval_path,
        {
            "status": "complete",
            "method": uncertainty["method"],
            "resamples": resamples,
            "seed": seed,
            "subset_oof_all_models": subset_intervals,
            "locked_hf_lw_gam": locked_intervals,
        },
    )
    summary_path = RESULT_ROOT / "cohort_bag_sensitivity_summary_v1.json"
    atomic_write_json(
        summary_path,
        {
            "status": "complete",
            "scope": "frozen_no_retraining_cohort_and_bag_sensitivity",
            "subset_model_count": len(MODEL_ORDER),
            "subset_cohort_counts": {
                "primary_234": 234,
                "scans_ge_2_218": 218,
                "behavior_sum_equal_scans_121": 121,
            },
            "locked_hf_cohort_counts": {
                "primary_234": 234,
                "numeric_behavior_236": 236,
                "scans_ge_2_218": 218,
                "behavior_sum_equal_scans_121": 121,
            },
            "bag_instance_counts": {
                "main_discrete_4126": 4126,
                "all_segments_4152": 4152,
                "high_confidence_3884": 3884,
            },
            "prediction_row_count": len(prediction_rows),
            "full_bag_reproduction_tolerance": 1.0e-7,
            "full_bag_max_absolute_difference_from_official_oof": full_reproduction_difference,
            "source_sha256": {
                str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
                for path in source_paths
            },
            "artifact_sha256": {
                str(path.relative_to(RESULT_ROOT)): sha256_file(path)
                for path in (prediction_path, metric_path, interval_path)
            },
            "interpretation_guard": execution["interpretation"]["guard"],
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "prediction_rows": len(prediction_rows),
                "full_bag_max_absolute_difference": full_reproduction_difference,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
