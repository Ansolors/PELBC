#!/usr/bin/env python3
"""Evaluate chronological whistle-prefix evidence with locked HF-LW-GAM models."""

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
BUDGETS: tuple[tuple[str, int | None], ...] = (
    ("first_1", 1),
    ("first_3", 3),
    ("first_5", 5),
    ("first_10", 10),
    ("first_20", 20),
    ("full_bag", None),
)
RESULT_ROOT = PROJECT_ROOT / "results/prefix_evaluation"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class PrefixMILBagDataset(MILBagDataset):
    """Evaluation-only bag view retaining the first min(k, N) instances."""

    def __init__(self, *args: Any, requested_instances: int | None, **kwargs: Any) -> None:
        if requested_instances is not None and int(requested_instances) < 1:
            raise ValueError("requested prefix size must be positive")
        self.requested_instances = (
            None if requested_instances is None else int(requested_instances)
        )
        super().__init__(*args, training=False, **kwargs)

    def _selected_whistles(self, encounter_id: str) -> tuple[Mapping[str, Any], ...]:
        values = self.registry.whistles[encounter_id]
        sequence = tuple(int(row["sequence_index"]) for row in values)
        if sequence != tuple(sorted(sequence)) or len(sequence) != len(set(sequence)):
            raise RuntimeError(f"non-chronological whistle registry: {encounter_id}")
        if self.requested_instances is None:
            return values
        return values[: self.requested_instances]


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


def interval(values: np.ndarray) -> dict[str, float]:
    lower, upper = percentile_interval(values, confidence_level=0.95)
    return {"lower": float(lower), "upper": float(upper)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(int(args.torch_threads))
    device = torch.device(str(args.device))
    modeling = load_toml(PROJECT_ROOT / "configs/modeling_protocol_v1.toml")
    validation = load_toml(PROJECT_ROOT / "configs/validation_protocol_v1.toml")
    registry = DatasetRegistry.load(
        PROJECT_ROOT,
        cache_features_in_memory=True,
    )
    registry.preload_feature_values()
    if len(registry.encounter_ids) != 234:
        raise RuntimeError("prefix evaluation requires the complete primary cohort")

    rows_by_budget: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_paths: list[Path] = [
        PROJECT_ROOT / "data/processed/v0.4.0/prefix_evaluation_registry.json",
        PROJECT_ROOT / "configs/modeling_protocol_v1.toml",
        PROJECT_ROOT / "configs/validation_protocol_v1.toml",
        PROJECT_ROOT / "scripts/run_prefix_evaluation.py",
        PROJECT_ROOT / "src/dolphin_behavior_mil/mil_models.py",
        PROJECT_ROOT / "src/dolphin_behavior_mil/modeling_data.py",
        PROJECT_ROOT / "src/dolphin_behavior_mil/nested_validation.py",
        PROJECT_ROOT / "src/dolphin_behavior_mil/statistical_analysis.py",
        PROJECT_ROOT / "src/dolphin_behavior_mil/training.py",
    ]
    for outer_fold_id in OUTER_FOLDS:
        selection_path = (
            PROJECT_ROOT
            / "results/nested_cv/selection"
            / MODEL_ID
            / outer_fold_id
            / "candidate_selection.json"
        )
        result_path = (
            PROJECT_ROOT
            / "results/nested_cv/outer_folds"
            / MODEL_ID
            / outer_fold_id
            / "outer_fold_result.json"
        )
        selection = read_json(selection_path)
        result = read_json(result_path)
        if selection.get("status") != "complete" or result.get("status") != "complete":
            raise RuntimeError(f"official result is incomplete: {outer_fold_id}")
        candidate = dict(selection["selected_candidate"]["candidate"])
        candidate_id = str(candidate["candidate_id"])
        if candidate_id != str(result["selected_candidate_id"]):
            raise RuntimeError(f"selection/result candidate mismatch: {outer_fold_id}")
        selection_sha = sha256_file(selection_path)
        if selection_sha != result["selection_artifact_sha256"]:
            raise RuntimeError(f"selection hash mismatch: {outer_fold_id}")
        _, test_ids = registry.outer_split(outer_fold_id)
        run_dirs = [
            official_run_directory(
                outer_fold_id,
                candidate_id,
                int(seed),
                selection_sha,
            )
            for seed in result["final_seeds"]
        ]
        normalizer_records = [read_json(path / "normalizer.json") for path in run_dirs]
        if any(record != normalizer_records[0] for record in normalizer_records[1:]):
            raise RuntimeError(f"seed normalizers differ: {outer_fold_id}")
        normalizer = FeatureNormalizer.from_dict(normalizer_records[0])
        normalized = NormalizedFeatureCache.build(registry, normalizer, test_ids)
        prefix_datasets = {
            budget_id: PrefixMILBagDataset(
                registry,
                test_ids,
                normalizer,
                requested_instances=requested,
                normalized_feature_cache=normalized,
            )
            for budget_id, requested in BUDGETS
        }
        probability_by_budget: dict[str, list[np.ndarray]] = defaultdict(list)
        reference: dict[str, Any] = {}
        for run_dir in run_dirs:
            checkpoint = torch.load(
                run_dir / "best_checkpoint.pt",
                map_location="cpu",
                weights_only=True,
            )
            if (
                checkpoint.get("model_id") != MODEL_ID
                or tuple(checkpoint.get("label_order", ())) != LABEL_ORDER
            ):
                raise RuntimeError(f"checkpoint contract mismatch: {run_dir.name}")
            model = build_model(
                candidate,
                int(modeling["neural_training"]["encoder_microbatch_max_instances"]),
            )
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.to(device)
            for budget_id, _ in BUDGETS:
                prediction = predict_mil(
                    model,
                    prefix_datasets[budget_id],
                    batch_size=int(candidate["encounter_batch_size"]),
                    device=device,
                    data_loader_workers=0,
                )
                if tuple(prediction.encounter_ids) != tuple(sorted(test_ids)):
                    raise RuntimeError(
                        f"prefix prediction partition mismatch: {outer_fold_id}/{budget_id}"
                    )
                probability_by_budget[budget_id].append(prediction.probabilities)
                reference[budget_id] = prediction
            del model
        calibrator_path = selection_path.parent / "calibrator.json"
        calibrator = read_json(calibrator_path)
        for budget_id, requested in BUDGETS:
            raw = np.mean(np.stack(probability_by_budget[budget_id]), axis=0)
            calibrated = apply_calibrator(raw, calibrator)
            prediction = reference[budget_id]
            for index, encounter_id in enumerate(prediction.encounter_ids):
                available = len(registry.whistles[encounter_id])
                used = available if requested is None else min(int(requested), available)
                rows_by_budget[budget_id].append(
                    {
                        "model_id": MODEL_ID,
                        "budget_id": budget_id,
                        "requested_whistles": requested,
                        "used_whistles": used,
                        "available_whistles": available,
                        "reached_requested_budget": requested is None or available >= requested,
                        "outer_fold_id": outer_fold_id,
                        "encounter_id": encounter_id,
                        "campaign_id": prediction.campaign_ids[index],
                        "label_order": list(LABEL_ORDER),
                        "y_true": prediction.labels[index].astype(int).tolist(),
                        "seed_probabilities": [
                            matrix[index].astype(float).tolist()
                            for matrix in probability_by_budget[budget_id]
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
        print(f"prefix inference complete: {outer_fold_id}", flush=True)

    ordered_rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    all_ids = tuple(sorted(registry.encounter_ids))
    canonical_campaigns: tuple[str, ...] | None = None
    canonical_truth: np.ndarray | None = None
    raw_by_budget: dict[str, np.ndarray] = {}
    for budget_id, requested in BUDGETS:
        rows = sorted(rows_by_budget[budget_id], key=lambda row: row["encounter_id"])
        if tuple(row["encounter_id"] for row in rows) != all_ids:
            raise RuntimeError(f"prefix OOF coverage mismatch: {budget_id}")
        truth = np.asarray([row["y_true"] for row in rows], dtype=np.int8)
        campaigns = tuple(str(row["campaign_id"]) for row in rows)
        raw = np.asarray([row["raw_probability"] for row in rows], dtype=np.float64)
        calibrated = np.asarray(
            [row["calibrated_probability"] for row in rows], dtype=np.float64
        )
        if canonical_truth is None:
            canonical_truth = truth
            canonical_campaigns = campaigns
        elif not np.array_equal(truth, canonical_truth) or campaigns != canonical_campaigns:
            raise RuntimeError("cross-budget labels or campaigns changed")
        raw_by_budget[budget_id] = raw
        all_metrics = {
            "encounter_count": len(rows),
            "reached_requested_budget_count": sum(
                bool(row["reached_requested_budget"]) for row in rows
            ),
            "raw": probability_metrics(truth, raw, LABEL_ORDER),
            "calibrated": probability_metrics(truth, calibrated, LABEL_ORDER),
        }
        strict_indices = np.flatnonzero(
            np.asarray([bool(row["reached_requested_budget"]) for row in rows])
        )
        strict_truth = truth[strict_indices]
        strict_raw = raw[strict_indices]
        all_metrics["strict_fixed_k_subset"] = {
            "encounter_count": int(len(strict_indices)),
            "campaign_count": len({campaigns[index] for index in strict_indices}),
            "positive_counts": {
                label: int(np.sum(strict_truth[:, label_index]))
                for label_index, label in enumerate(LABEL_ORDER)
            },
            "prevalence": {
                label: float(np.mean(strict_truth[:, label_index]))
                for label_index, label in enumerate(LABEL_ORDER)
            },
            "raw": probability_metrics(strict_truth, strict_raw, LABEL_ORDER),
        }
        metrics[budget_id] = all_metrics
        ordered_rows.extend(rows)

    assert canonical_truth is not None and canonical_campaigns is not None
    official = sorted(
        read_jsonl(
            PROJECT_ROOT
            / "results/nested_cv/oof"
            / MODEL_ID
            / "oof_predictions.jsonl"
        ),
        key=lambda row: row["encounter_id"],
    )
    official_raw = np.asarray(
        [row["raw_probability"] for row in official], dtype=np.float64
    )
    full_difference = float(np.max(np.abs(raw_by_budget["full_bag"] - official_raw)))
    if full_difference > 1.0e-7:
        raise RuntimeError(
            f"reloaded full-bag predictions differ from official OOF: {full_difference}"
        )

    uncertainty = validation["uncertainty"]
    plan = draw_valid_campaign_bootstrap_plan(
        canonical_truth,
        canonical_campaigns,
        resamples=int(uncertainty["resamples"]),
        seed=int(uncertainty["seed"]),
    )
    bootstrap: dict[str, Any] = {}
    for budget_id, _ in BUDGETS:
        macro, per_label = bootstrap_average_precision(
            canonical_truth,
            raw_by_budget[budget_id],
            plan,
        )
        bootstrap[budget_id] = {
            "macro_average_precision": {
                "point_estimate": metrics[budget_id]["raw"][
                    "macro_average_precision"
                ],
                **interval(macro),
            },
            "per_label_average_precision": {
                label: {
                    "point_estimate": metrics[budget_id]["raw"][
                        "per_label_average_precision"
                    ][label],
                    **interval(per_label[:, label_index]),
                }
                for label_index, label in enumerate(LABEL_ORDER)
            },
        }
        print(f"prefix bootstrap complete: {budget_id}", flush=True)

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    prediction_path = RESULT_ROOT / "hf_lw_gam_prefix_predictions_v1.jsonl"
    metric_path = RESULT_ROOT / "prefix_metrics_v1.json"
    bootstrap_path = RESULT_ROOT / "prefix_bootstrap_intervals_v1.json"
    atomic_write_jsonl(prediction_path, ordered_rows)
    atomic_write_json(
        metric_path,
        {
            "status": "complete",
            "model_id": MODEL_ID,
            "analysis_unit": "encounter",
            "label_scope": "encounter_bag",
            "label_order": list(LABEL_ORDER),
            "budget_order": [budget_id for budget_id, _ in BUDGETS],
            "metrics": metrics,
            "full_bag_max_absolute_difference_from_official_oof": full_difference,
        },
    )
    atomic_write_json(
        bootstrap_path,
        {
            "method": uncertainty["method"],
            "resamples": plan.resamples,
            "seed": plan.seed,
            "campaign_count": len(plan.campaign_order),
            "intervals": bootstrap,
        },
    )
    summary_path = RESULT_ROOT / "prefix_evaluation_summary_v1.json"
    atomic_write_json(
        summary_path,
        {
            "status": "complete",
            "model_id": MODEL_ID,
            "encounter_count_per_budget": 234,
            "budget_count": len(BUDGETS),
            "prediction_row_count": len(ordered_rows),
            "full_bag_reproduction_tolerance": 1.0e-7,
            "full_bag_max_absolute_difference_from_official_oof": full_difference,
            "source_sha256": {
                str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
                for path in source_paths
            },
            "artifact_sha256": {
                str(path.relative_to(RESULT_ROOT)): sha256_file(path)
                for path in (prediction_path, metric_path, bootstrap_path)
            },
            "claim_boundary": (
                "offline evidence accumulation from archived detected whistles; not "
                "validated hydrophone detection latency, segmentation, or at-sea behavior"
            ),
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "budgets": len(BUDGETS),
                "prediction_rows": len(ordered_rows),
                "full_bag_max_absolute_difference": full_difference,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
