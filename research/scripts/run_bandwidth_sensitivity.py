#!/usr/bin/env python3
"""Train the paired locked-configuration 192-kHz HF-LW-GAM sensitivity."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dolphin_behavior_mil.bandwidth_data import (  # noqa: E402
    BandwidthFeatureRegistry,
    MelFeatureNormalizer,
    build_bandwidth_hf_lw_gam,
)
from dolphin_behavior_mil.evaluation import percentile_interval  # noqa: E402
from dolphin_behavior_mil.experiment import (  # noqa: E402
    RunLedger,
    atomic_write_json,
    atomic_write_jsonl,
    load_toml,
    run_identity,
    sha256_file,
)
from dolphin_behavior_mil.modeling_data import (  # noqa: E402
    LABEL_ORDER,
    MILBagDataset,
    NormalizedFeatureCache,
    partition_sha256,
)
from dolphin_behavior_mil.statistical_analysis import (  # noqa: E402
    bootstrap_average_precision,
    draw_valid_campaign_bootstrap_plan,
    probability_metrics,
)
from dolphin_behavior_mil.training import (  # noqa: E402
    NeuralTrainingConfig,
    dataset_preprocessing_record,
    fit_neural_fixed_epochs,
    predict_mil,
    seed_everything,
)


MODEL_ID = "hf_lw_gam"
RUN_MODEL_ID = "hf_lw_gam_192k"
STAGE = "bandwidth_sensitivity_final_v1"
OUTER_FOLDS = tuple(f"outer_{index:02d}" for index in range(1, 6))
MANIFEST_PATH = (
    PROJECT_ROOT / "results/sensitivity/bandwidth/logmel_192k_manifest_v1.jsonl"
)
CACHE_SUMMARY_PATH = (
    PROJECT_ROOT / "results/sensitivity/bandwidth/logmel_192k_summary_v1.json"
)
OUTPUT_ROOT = PROJECT_ROOT / "results/sensitivity/bandwidth"
RUN_ROOT = PROJECT_ROOT / "results/model_runs"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_model(candidate: Mapping[str, Any], modeling: Mapping[str, Any]) -> Any:
    return build_bandwidth_hf_lw_gam(
        mel_count=192,
        base_channels=int(candidate["base_channels"]),
        embedding_dim=int(candidate["embedding_dim"]),
        attention_dim=int(candidate["attention_dim"]),
        dropout=float(candidate["dropout"]),
        encoder_microbatch_max_instances=int(
            modeling["neural_training"]["encoder_microbatch_max_instances"]
        ),
    )


def training_config(
    candidate: Mapping[str, Any],
    modeling: Mapping[str, Any],
    *,
    seed: int,
    epochs: int,
) -> NeuralTrainingConfig:
    common = modeling["neural_training"]
    return NeuralTrainingConfig(
        learning_rate=float(candidate["learning_rate"]),
        weight_decay=float(candidate["weight_decay"]),
        encounter_batch_size=int(candidate["encounter_batch_size"]),
        maximum_epochs=int(epochs),
        early_stopping_patience=min(int(common["early_stopping_patience"]), int(epochs)),
        early_stopping_minimum_delta=float(common["early_stopping_minimum_delta"]),
        gradient_clip_norm=float(common["gradient_clip_norm"]),
        seed=int(seed),
        deterministic_algorithms=bool(common["deterministic_algorithms"]),
        data_loader_workers=0,
    )


def interval(values: np.ndarray) -> dict[str, float]:
    lower, upper = percentile_interval(values, confidence_level=0.95)
    return {"lower": float(lower), "upper": float(upper)}


def reusable_run(
    *,
    outer_fold_id: str,
    candidate_id: str,
    seed: int,
    configuration: Mapping[str, Any],
    train_ids: tuple[str, ...],
    test_ids: tuple[str, ...],
    source_hashes: Mapping[str, str],
) -> tuple[Path, list[dict[str, Any]]] | None:
    run_id, _ = run_identity(
        stage=STAGE,
        model_id=RUN_MODEL_ID,
        outer_fold_id=outer_fold_id,
        inner_fold_id=None,
        candidate_id=candidate_id,
        seed=seed,
    )
    run_dir = RUN_ROOT / run_id
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise RuntimeError(f"incomplete deterministic bandwidth run exists: {run_id}")
    if (
        read_json(run_dir / "configuration.json") != dict(configuration)
        or manifest.get("train_partition_sha256") != partition_sha256(train_ids)
        or manifest.get("validation_partition_sha256") != partition_sha256(test_ids)
        or manifest.get("source_hashes") != dict(source_hashes)
        or manifest.get("outer_test_accessed") is not True
    ):
        raise RuntimeError(f"completed bandwidth run contract changed: {run_id}")
    artifact_hashes = manifest.get("artifact_sha256", {})
    if not artifact_hashes or not all(
        (run_dir / name).is_file() and sha256_file(run_dir / name) == digest
        for name, digest in artifact_hashes.items()
    ):
        raise RuntimeError(f"bandwidth run artifact hash mismatch: {run_id}")
    return run_dir, read_jsonl(run_dir / "predictions.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(int(args.torch_threads))
    device = torch.device(str(args.device))
    execution_path = PROJECT_ROOT / "configs/bandwidth_sensitivity_execution_v1.toml"
    execution = load_toml(execution_path)
    modeling = load_toml(PROJECT_ROOT / "configs/modeling_protocol_v1.toml")
    validation = load_toml(PROJECT_ROOT / "configs/validation_protocol_v1.toml")
    cache_summary = read_json(CACHE_SUMMARY_PATH)
    if (
        cache_summary.get("status") != "complete"
        or int(cache_summary.get("feature_count", -1)) != 4126
        or sha256_file(MANIFEST_PATH) != cache_summary["manifest_sha256"]
    ):
        raise RuntimeError("complete verified 192-kHz cache is required")
    registry = BandwidthFeatureRegistry.load(
        PROJECT_ROOT,
        MANIFEST_PATH,
        mel_count=int(execution["n_mels"]),
        verify_files=True,
        cache_features_in_memory=True,
    )
    fold_rows: list[dict[str, Any]] = []
    fold_results: dict[str, Any] = {}
    source_paths: list[Path] = [
        execution_path,
        MANIFEST_PATH,
        CACHE_SUMMARY_PATH,
        PROJECT_ROOT / "configs/modeling_protocol_v1.toml",
        PROJECT_ROOT / "configs/validation_protocol_v1.toml",
        PROJECT_ROOT / "scripts/run_bandwidth_sensitivity.py",
        PROJECT_ROOT / "src/dolphin_behavior_mil/bandwidth_data.py",
        PROJECT_ROOT / "src/dolphin_behavior_mil/mil_models.py",
        PROJECT_ROOT / "src/dolphin_behavior_mil/modeling_data.py",
        PROJECT_ROOT / "src/dolphin_behavior_mil/training.py",
    ]
    for outer_fold_id in OUTER_FOLDS:
        selection_path = (
            PROJECT_ROOT
            / "results/nested_cv/selection/hf_lw_gam"
            / outer_fold_id
            / "candidate_selection.json"
        )
        primary_result_path = (
            PROJECT_ROOT
            / "results/nested_cv/outer_folds/hf_lw_gam"
            / outer_fold_id
            / "outer_fold_result.json"
        )
        selection = read_json(selection_path)
        primary_result = read_json(primary_result_path)
        candidate = dict(selection["selected_candidate"]["candidate"])
        candidate_id = str(candidate["candidate_id"])
        fixed_epochs = int(primary_result["fixed_epochs"])
        selection_sha = sha256_file(selection_path)
        if (
            candidate_id != str(primary_result["selected_candidate_id"])
            or selection_sha != str(primary_result["selection_artifact_sha256"])
            or fixed_epochs
            != int(selection["selected_candidate"]["selected_final_epoch_if_chosen"])
        ):
            raise RuntimeError(f"primary pairing contract mismatch: {outer_fold_id}")
        train_ids, test_ids = registry.outer_split(outer_fold_id)
        normalizer = MelFeatureNormalizer.fit(registry, train_ids)
        normalized = NormalizedFeatureCache.build(
            registry, normalizer, tuple(sorted(set(train_ids) | set(test_ids)))
        )
        source_hashes = {
            "bandwidth_execution_protocol": sha256_file(execution_path),
            "bandwidth_manifest": sha256_file(MANIFEST_PATH),
            "bandwidth_cache_summary": sha256_file(CACHE_SUMMARY_PATH),
            "modeling_protocol": sha256_file(
                PROJECT_ROOT / "configs/modeling_protocol_v1.toml"
            ),
            "validation_protocol": sha256_file(
                PROJECT_ROOT / "configs/validation_protocol_v1.toml"
            ),
            "primary_selection": selection_sha,
            "primary_outer_result": sha256_file(primary_result_path),
            "bandwidth_runner": sha256_file(Path(__file__)),
            "bandwidth_data": sha256_file(
                PROJECT_ROOT / "src/dolphin_behavior_mil/bandwidth_data.py"
            ),
        }
        matrices: list[np.ndarray] = []
        seed_run_ids: list[str] = []
        reference_rows: list[dict[str, Any]] | None = None
        for seed in tuple(int(value) for value in execution["paired_training"]["final_seeds"]):
            config = training_config(
                candidate, modeling, seed=seed, epochs=fixed_epochs
            )
            configuration = {
                "scope": "paired_192khz_bandwidth_sensitivity_final_test",
                "representation_id": execution["sensitivity_representation_id"],
                "model_id": MODEL_ID,
                "run_model_id": RUN_MODEL_ID,
                "outer_fold_id": outer_fold_id,
                "candidate": candidate,
                "candidate_source": "official_96khz_inner_selection",
                "fixed_epochs": fixed_epochs,
                "fixed_epoch_source": "official_96khz_inner_selection",
                "seed": seed,
                "training": config.__dict__,
                "device": str(device),
                "outer_test_accessed": True,
            }
            reusable = reusable_run(
                outer_fold_id=outer_fold_id,
                candidate_id=candidate_id,
                seed=seed,
                configuration=configuration,
                train_ids=train_ids,
                test_ids=test_ids,
                source_hashes=source_hashes,
            )
            if reusable is not None:
                run_dir, prediction_rows = reusable
                print(f"[bandwidth resume] {run_dir.name}", flush=True)
            else:
                ledger = RunLedger.initialize(
                    RUN_ROOT,
                    stage=STAGE,
                    model_id=RUN_MODEL_ID,
                    outer_fold_id=outer_fold_id,
                    inner_fold_id=None,
                    candidate_id=candidate_id,
                    seed=seed,
                    configuration=configuration,
                    train_encounter_ids=train_ids,
                    validation_encounter_ids=test_ids,
                    source_hashes=source_hashes,
                )
                seed_everything(seed, deterministic_algorithms=True)
                train_dataset = MILBagDataset(
                    registry,
                    train_ids,
                    normalizer,
                    training=True,
                    maximum_instances_per_bag=int(
                        candidate["maximum_training_instances_per_bag"]
                    ),
                    seed=seed,
                    augmentation=modeling["augmentation"][str(candidate["augmentation"])],
                    normalized_feature_cache=normalized,
                )
                test_dataset = MILBagDataset(
                    registry,
                    test_ids,
                    normalizer,
                    training=False,
                    normalized_feature_cache=normalized,
                )
                preprocessing = dataset_preprocessing_record(
                    train_dataset, test_dataset
                )
                model = build_model(candidate, modeling)
                fit = fit_neural_fixed_epochs(
                    model,
                    train_dataset,
                    config,
                    epochs=fixed_epochs,
                    device=device,
                )
                prediction = predict_mil(
                    model,
                    test_dataset,
                    batch_size=config.encounter_batch_size,
                    device=device,
                    data_loader_workers=0,
                )
                ledger.save_fixed_epoch_result(
                    fit,
                    prediction,
                    model_id=RUN_MODEL_ID,
                    preprocessing=preprocessing,
                    selection_artifact_sha256=selection_sha,
                )
                run_dir = ledger.run_dir
                prediction_rows = read_jsonl(run_dir / "predictions.jsonl")
                del model, train_dataset, test_dataset, fit, prediction
                gc.collect()
                print(f"[bandwidth complete] {run_dir.name}", flush=True)
            ordered = sorted(prediction_rows, key=lambda row: row["encounter_id"])
            if tuple(row["encounter_id"] for row in ordered) != tuple(sorted(test_ids)):
                raise RuntimeError(f"bandwidth test coverage mismatch: {run_dir.name}")
            matrices.append(
                np.asarray([row["probability"] for row in ordered], dtype=np.float64)
            )
            reference_rows = ordered
            seed_run_ids.append(run_dir.name)
        assert reference_rows is not None
        raw = np.mean(np.stack(matrices), axis=0)
        truth = np.asarray([row["y_true"] for row in reference_rows], dtype=np.int8)
        for index, row in enumerate(reference_rows):
            fold_rows.append(
                {
                    "model_id": RUN_MODEL_ID,
                    "representation_id": execution["sensitivity_representation_id"],
                    "outer_fold_id": outer_fold_id,
                    "encounter_id": str(row["encounter_id"]),
                    "campaign_id": str(row["campaign_id"]),
                    "label_order": list(LABEL_ORDER),
                    "y_true": truth[index].astype(int).tolist(),
                    "seed_probabilities": [
                        matrix[index].astype(float).tolist() for matrix in matrices
                    ],
                    "raw_probability": raw[index].astype(float).tolist(),
                    "label_scope": "encounter_bag",
                }
            )
        fold_result = {
            "status": "complete",
            "outer_fold_id": outer_fold_id,
            "train_count": len(train_ids),
            "test_count": len(test_ids),
            "selected_candidate_id": candidate_id,
            "fixed_epochs": fixed_epochs,
            "final_seeds": [int(value) for value in execution["paired_training"]["final_seeds"]],
            "run_ids": seed_run_ids,
            "raw_metrics": probability_metrics(truth, raw, LABEL_ORDER),
            "source_hashes": source_hashes,
        }
        fold_results[outer_fold_id] = fold_result
        fold_output = OUTPUT_ROOT / "outer_folds" / outer_fold_id
        fold_output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(fold_output / "outer_fold_result.json", fold_result)
        print(
            f"bandwidth fold complete: {outer_fold_id} raw AP="
            f"{fold_result['raw_metrics']['macro_average_precision']:.6f}",
            flush=True,
        )
        source_paths.extend([selection_path, primary_result_path])

    rows = sorted(fold_rows, key=lambda row: row["encounter_id"])
    if len(rows) != 234 or len({row["encounter_id"] for row in rows}) != 234:
        raise RuntimeError("bandwidth OOF does not cover 234 unique encounters")
    truth = np.asarray([row["y_true"] for row in rows], dtype=np.int8)
    probability_192 = np.asarray(
        [row["raw_probability"] for row in rows], dtype=np.float64
    )
    campaigns = tuple(str(row["campaign_id"]) for row in rows)
    primary_rows = sorted(
        read_jsonl(
            PROJECT_ROOT / "results/nested_cv/oof/hf_lw_gam/oof_predictions.jsonl"
        ),
        key=lambda row: row["encounter_id"],
    )
    if tuple(row["encounter_id"] for row in primary_rows) != tuple(
        row["encounter_id"] for row in rows
    ):
        raise RuntimeError("96/192-kHz OOF encounter alignment mismatch")
    probability_96 = np.asarray(
        [row["raw_probability"] for row in primary_rows], dtype=np.float64
    )
    metrics_192 = probability_metrics(truth, probability_192, LABEL_ORDER)
    metrics_96 = probability_metrics(truth, probability_96, LABEL_ORDER)
    uncertainty = validation["uncertainty"]
    plan = draw_valid_campaign_bootstrap_plan(
        truth,
        campaigns,
        resamples=int(uncertainty["resamples"]),
        seed=int(uncertainty["seed"]),
    )
    macro_192, labels_192 = bootstrap_average_precision(truth, probability_192, plan)
    macro_96, labels_96 = bootstrap_average_precision(truth, probability_96, plan)
    delta_macro = macro_192 - macro_96
    delta_labels = labels_192 - labels_96
    comparison = {
        "status": "complete",
        "scope": "paired_locked_configuration_bandwidth_sensitivity",
        "encounter_count": 234,
        "campaign_count": len(plan.campaign_order),
        "resamples": plan.resamples,
        "representation_96khz": {
            "id": execution["main_representation_id"],
            "metrics": metrics_96,
            "macro_ap_interval": {
                "point_estimate": metrics_96["macro_average_precision"],
                **interval(macro_96),
            },
        },
        "representation_192khz": {
            "id": execution["sensitivity_representation_id"],
            "metrics": metrics_192,
            "macro_ap_interval": {
                "point_estimate": metrics_192["macro_average_precision"],
                **interval(macro_192),
            },
        },
        "paired_delta_192_minus_96": {
            "macro_average_precision": {
                "point_estimate": float(
                    metrics_192["macro_average_precision"]
                    - metrics_96["macro_average_precision"]
                ),
                **interval(delta_macro),
            },
            "per_label_average_precision": {
                label: {
                    "point_estimate": float(
                        metrics_192["per_label_average_precision"][label]
                        - metrics_96["per_label_average_precision"][label]
                    ),
                    **interval(delta_labels[:, index]),
                }
                for index, label in enumerate(LABEL_ORDER)
            },
        },
        "interpretation_guard": execution["interpretation"]["guard"],
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    prediction_path = OUTPUT_ROOT / "hf_lw_gam_192k_oof_predictions_v1.jsonl"
    comparison_path = OUTPUT_ROOT / "bandwidth_comparison_v1.json"
    atomic_write_jsonl(prediction_path, rows)
    atomic_write_json(comparison_path, comparison)
    summary_path = OUTPUT_ROOT / "bandwidth_sensitivity_summary_v1.json"
    atomic_write_json(
        summary_path,
        {
            "status": "complete",
            "model_id": MODEL_ID,
            "run_model_id": RUN_MODEL_ID,
            "outer_fold_count": len(OUTER_FOLDS),
            "final_run_count": sum(len(value["run_ids"]) for value in fold_results.values()),
            "oof_encounter_count": len(rows),
            "paired_delta_192_minus_96_macro_ap": comparison[
                "paired_delta_192_minus_96"
            ]["macro_average_precision"],
            "source_sha256": {
                str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
                for path in source_paths
            },
            "artifact_sha256": {
                str(path.relative_to(OUTPUT_ROOT)): sha256_file(path)
                for path in (prediction_path, comparison_path)
            },
            "interpretation_guard": execution["interpretation"]["guard"],
        },
    )
    print(json.dumps(comparison["paired_delta_192_minus_96"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
