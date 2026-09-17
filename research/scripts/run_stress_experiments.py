#!/usr/bin/env python3
"""Run resumable registered domain-shift stress experiments."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_nested_experiments import NestedRunner  # noqa: E402
from dolphin_behavior_mil.classical import fit_classical_baseline  # noqa: E402
from dolphin_behavior_mil.evaluation import evaluation_record  # noqa: E402
from dolphin_behavior_mil.experiment import (  # noqa: E402
    MODEL_FAMILIES,
    RunLedger,
    atomic_write_json,
    atomic_write_jsonl,
    load_toml,
    sha256_file,
)
from dolphin_behavior_mil.modeling_data import LABEL_ORDER, partition_sha256  # noqa: E402
from dolphin_behavior_mil.nested_validation import (  # noqa: E402
    CLASSICAL_MODEL_IDS,
    CNN_MODEL_IDS,
    PRETRAINED_MODEL_IDS,
    apply_calibrator,
    assemble_oof_rows,
    candidates_for_model,
    ensemble_prediction_matrices,
    fit_platt_calibrator,
    identity_calibrator,
    select_candidate,
    select_multilabel_thresholds,
    summarize_candidate,
)
from dolphin_behavior_mil.statistical_analysis import probability_metrics  # noqa: E402
from dolphin_behavior_mil.stress_validation import StressScenarioRegistry  # noqa: E402
from dolphin_behavior_mil.training import (  # noqa: E402
    dataset_preprocessing_record,
    fit_neural_fixed_epochs,
    fit_neural_model,
    predict_mil,
    seed_everything,
)


MODEL_ORDER = (
    "prior_constant",
    "bag_size_logistic",
    "metadata_logistic",
    "handcrafted_logistic",
    "hf_lw_gam",
    "cnn_mean",
    "panns_frozen_mil",
    "aves_frozen_mil",
)
MATCHED_STRESS_ABLATIONS = frozenset({"cnn_mean"})
SELECTION_STAGE = "stress_inner_selection_v1"
TEST_STAGE = "stress_final_test_v1"
OUTPUT_ROOT = PROJECT_ROOT / "results/stress_tests"
RUN_ROOT = PROJECT_ROOT / "results/model_runs"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selection_dir(model_id: str, scenario_id: str) -> Path:
    return OUTPUT_ROOT / "selection" / model_id / scenario_id


def test_dir(model_id: str, scenario_id: str) -> Path:
    return OUTPUT_ROOT / "test" / model_id / scenario_id


def scenario_group(scenario_id: str) -> str:
    if scenario_id.startswith("leave_year_"):
        return "leave_year_out"
    if scenario_id.startswith("leave_location_"):
        return "leave_location_out"
    if scenario_id.startswith("forward_train_"):
        return "strict_forward"
    if scenario_id == "train_288khz_test_384_or_576khz":
        return "sample_rate_shift"
    raise KeyError(scenario_id)


def configured_pairs(
    execution: Mapping[str, Any],
    registry: StressScenarioRegistry,
) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for scenario_id in registry.scenario_ids:
        group = scenario_group(scenario_id)
        models = tuple(str(value) for value in execution["scope"][group])
        unknown = sorted(set(models) - set(MODEL_ORDER))
        if unknown:
            raise ValueError(f"unknown stress models in {group}: {unknown}")
        for model_id in MODEL_ORDER:
            if model_id in models:
                pairs.append((model_id, scenario_id))
    return tuple(pairs)


def parse_subset(value: str, allowed: Sequence[str], name: str) -> tuple[str, ...]:
    if value == "all":
        return tuple(allowed)
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(values) - set(allowed))
    if not values or unknown:
        raise ValueError(f"invalid {name}: {unknown or values}")
    return values


class RegisteredStressRunner:
    def __init__(
        self,
        *,
        models: Sequence[str],
        device: str,
        torch_threads: int,
        quiet_trials: bool,
    ) -> None:
        self.execution_path = PROJECT_ROOT / "configs/stress_execution_v1.toml"
        if not self.execution_path.is_file():
            raise FileNotFoundError(
                "stress execution scope must be frozen before running: "
                f"{self.execution_path}"
            )
        self.execution = load_toml(self.execution_path)
        self.scenarios = StressScenarioRegistry.load(PROJECT_ROOT)
        self.engine = NestedRunner(
            models=models,
            device=device,
            torch_threads=torch_threads,
            preload_logmel=False,
            quiet_trials=quiet_trials,
            encoder_microbatch_override=None,
        )
        self.device = str(device)
        self.quiet_trials = bool(quiet_trials)

    def source_hashes(self, model_id: str, scenario_id: str) -> dict[str, str]:
        paths = {
            "stress_execution_protocol": self.execution_path,
            "validation_protocol": PROJECT_ROOT / "configs/validation_protocol_v1.toml",
            "modeling_protocol": PROJECT_ROOT / "configs/modeling_protocol_v1.toml",
            "cnn_compute_protocol": PROJECT_ROOT / "configs/cnn_compute_protocol_v1.toml",
            "dataset_manifest": PROJECT_ROOT / "data/processed/v0.4.0/manifest.json",
            "temporal_assignments": PROJECT_ROOT
            / "data/processed/v0.4.0/temporal_stress_assignments.jsonl",
            "location_assignments": PROJECT_ROOT
            / "data/processed/v0.4.0/location_stress_assignments.jsonl",
            "stress_inner_assignments": PROJECT_ROOT
            / "data/processed/v0.4.0/stress_inner_fold_assignments.jsonl",
            "stress_runner": PROJECT_ROOT / "scripts/run_stress_experiments.py",
            "stress_registry": PROJECT_ROOT
            / "src/dolphin_behavior_mil/stress_validation.py",
        }
        if model_id in PRETRAINED_MODEL_IDS:
            key = "panns" if model_id == "panns_frozen_mil" else "aves"
            paths["pretrained_protocol"] = PROJECT_ROOT / "configs/pretrained_encoders_v1.toml"
            paths["embedding_manifest"] = PROJECT_ROOT / str(
                self.engine.pretrained["encoders"][key]["manifest_path"]
            )
        if model_id in MATCHED_STRESS_ABLATIONS:
            dependency = (
                selection_dir("hf_lw_gam", scenario_id) / "candidate_selection.json"
            )
            if not dependency.is_file():
                raise FileNotFoundError(
                    f"HF-LW-GAM stress selection must precede {model_id}: {dependency}"
                )
            paths["matched_hf_lw_gam_stress_selection"] = dependency
        return {name: sha256_file(path) for name, path in paths.items()}

    def candidates(self, model_id: str, scenario_id: str) -> list[dict[str, Any]]:
        if model_id in MATCHED_STRESS_ABLATIONS:
            selection = read_json(
                selection_dir("hf_lw_gam", scenario_id) / "candidate_selection.json"
            )
            if selection.get("status") != "complete":
                raise RuntimeError("matched HF-LW-GAM stress selection is incomplete")
            return [dict(selection["selected_candidate"]["candidate"])]
        values = candidates_for_model(self.engine.modeling, model_id)
        if model_id == "hf_lw_gam":
            eligible = set(
                str(value)
                for value in self.engine.cnn_compute["candidate_subset"][
                    "eligible_candidate_ids"
                ]
            )
            values = [row for row in values if str(row["candidate_id"]) in eligible]
            if {str(row["candidate_id"]) for row in values} != eligible:
                raise RuntimeError("stress CNN candidate set differs from frozen overlay")
        return values

    def run_trial(
        self,
        model_id: str,
        scenario_id: str,
        fold_id: str,
        candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        train_ids, validation_ids = self.scenarios.selection_split(
            scenario_id, fold_id
        )
        seed = int(self.engine.validation["model_selection"]["selection_seed"])
        candidate_id = str(candidate["candidate_id"])
        source_hashes = self.source_hashes(model_id, scenario_id)
        training_config = (
            self.engine.neural_training_config(model_id, candidate, seed=seed)
            if model_id not in CLASSICAL_MODEL_IDS
            else None
        )
        configuration = {
            "scope": "registered_domain_shift_inner_selection",
            "scenario_id": scenario_id,
            "scenario_group": scenario_group(scenario_id),
            "model_id": model_id,
            "family": MODEL_FAMILIES[model_id],
            "fold_id": fold_id,
            "candidate": dict(candidate),
            "training": None if training_config is None else training_config.__dict__,
            "device": self.device,
            "outer_test_accessed": False,
        }
        reusable = self.engine.completed_run(
            stage=SELECTION_STAGE,
            model_id=model_id,
            outer_fold_id=scenario_id,
            inner_fold_id=fold_id,
            candidate_id=candidate_id,
            seed=seed,
            configuration=configuration,
            train_ids=train_ids,
            validation_ids=validation_ids,
            source_hashes=source_hashes,
        )
        if reusable is not None:
            run_dir, metrics, predictions = reusable
            if not self.quiet_trials:
                print(f"[stress resume] {run_dir.name}", flush=True)
            return self.engine.trial_record(
                run_dir, metrics, predictions, inner_fold_id=fold_id
            )
        ledger = RunLedger.initialize(
            RUN_ROOT,
            stage=SELECTION_STAGE,
            model_id=model_id,
            outer_fold_id=scenario_id,
            inner_fold_id=fold_id,
            candidate_id=candidate_id,
            seed=seed,
            configuration=configuration,
            train_encounter_ids=train_ids,
            validation_encounter_ids=validation_ids,
            source_hashes=source_hashes,
        )
        if not self.quiet_trials:
            print(f"[stress run] {ledger.run_id}", flush=True)
        if model_id in CLASSICAL_MODEL_IDS:
            fit = fit_classical_baseline(
                self.engine.registry,
                train_ids,
                validation_ids,
                model_id=model_id,
                modeling_config=self.engine.modeling,
                candidate=None if model_id == "prior_constant" else candidate,
                random_seed=seed,
            )
            ledger.save_classical_result(
                fit,
                model_id=model_id,
                outer_test_accessed=False,
            )
        else:
            assert training_config is not None
            seed_everything(seed, deterministic_algorithms=True)
            model = self.engine.build_model(model_id, candidate)
            train_dataset, validation_dataset, preprocessing = self.engine.build_datasets(
                model_id,
                candidate,
                train_ids,
                validation_ids,
                seed=seed,
            )
            fit = fit_neural_model(
                model,
                train_dataset,
                validation_dataset,
                training_config,
                device=self.device,
            )
            ledger.save_fit_result(
                fit,
                model_id=model_id,
                normalizer=preprocessing,
            )
            del model, train_dataset, validation_dataset, fit
            gc.collect()
        metrics = read_json(ledger.run_dir / "metrics.json")
        predictions = read_jsonl(ledger.run_dir / "predictions.jsonl")
        return self.engine.trial_record(
            ledger.run_dir, metrics, predictions, inner_fold_id=fold_id
        )

    def run_selection(self, model_id: str, scenario_id: str) -> dict[str, Any]:
        scenario = self.scenarios.scenario(scenario_id)
        fold_ids = self.scenarios.selection_fold_ids(scenario_id)
        candidates = self.candidates(model_id, scenario_id)
        records_by_candidate: dict[str, list[dict[str, Any]]] = {
            str(row["candidate_id"]): [] for row in candidates
        }
        for fold_id in fold_ids:
            for candidate in candidates:
                records_by_candidate[str(candidate["candidate_id"])].append(
                    self.run_trial(model_id, scenario_id, fold_id, candidate)
                )
        summaries = [
            summarize_candidate(
                candidate,
                records_by_candidate[str(candidate["candidate_id"])],
                expected_fold_count=len(fold_ids),
            )
            for candidate in candidates
        ]
        selected = select_candidate(
            summaries,
            tie_tolerance_macro_ap=float(
                self.engine.validation["model_selection"]["tie_tolerance_macro_ap"]
            ),
        )
        selected_records = records_by_candidate[str(selected["candidate_id"])]
        prediction_rows: list[dict[str, Any]] = []
        for record in selected_records:
            prediction_rows.extend(
                read_jsonl(
                    PROJECT_ROOT
                    / str(record["run_directory_relative_to_project"])
                    / "predictions.jsonl"
                )
            )
        expected_ids = (
            scenario.validation_ids
            if scenario.uses_fixed_forward_validation
            else scenario.train_ids
        )
        oof = assemble_oof_rows(prediction_rows, expected_ids)
        calibration = self.engine.validation["calibration"]
        if model_id == "prior_constant" and not bool(
            calibration["constant_prior_baseline_recalibration"]
        ):
            calibrator = identity_calibrator()
        else:
            calibrator = fit_platt_calibrator(
                oof["labels"],
                oof["probabilities"],
                c_value=float(calibration["l2_inverse_regularization_c"]),
                probability_clip=calibration["probability_clip"],
                random_seed=int(
                    self.engine.validation["model_selection"]["selection_seed"]
                ),
            )
        calibrated = apply_calibrator(oof["probabilities"], calibrator)
        thresholds = select_multilabel_thresholds(oof["labels"], calibrated)
        output = selection_dir(model_id, scenario_id)
        output.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "model_id": model_id,
                "scenario_id": scenario_id,
                "selection_fold_scope": (
                    "fixed_2024_validation"
                    if scenario.uses_fixed_forward_validation
                    else "campaign_grouped_inner_oof"
                ),
                "encounter_id": encounter_id,
                "campaign_id": oof["campaign_ids"][index],
                "label_order": list(LABEL_ORDER),
                "y_true": oof["labels"][index].astype(int).tolist(),
                "raw_probability": oof["probabilities"][index].astype(float).tolist(),
                "calibrated_probability": calibrated[index].astype(float).tolist(),
                "outer_test_accessed": False,
            }
            for index, encounter_id in enumerate(oof["encounter_ids"])
        ]
        prediction_path = output / "selected_validation_predictions.jsonl"
        calibrator_path = output / "calibrator.json"
        threshold_path = output / "thresholds.json"
        atomic_write_jsonl(prediction_path, rows)
        atomic_write_json(calibrator_path, calibrator)
        atomic_write_json(threshold_path, thresholds)
        selection = {
            "status": "complete",
            "scope": "registered_domain_shift_model_selection",
            "model_id": model_id,
            "scenario_id": scenario_id,
            "scenario_group": scenario_group(scenario_id),
            "selection_fold_ids": list(fold_ids),
            "selection_encounter_count": len(expected_ids),
            "selection_partition_sha256": partition_sha256(expected_ids),
            "candidate_count": len(candidates),
            "outer_test_accessed": False,
            "candidate_summaries": summaries,
            "selected_candidate": selected,
            "selected_validation_raw_metrics": evaluation_record(
                oof["labels"], oof["probabilities"], LABEL_ORDER
            ),
            "selected_validation_calibrated_metrics": evaluation_record(
                oof["labels"], calibrated, LABEL_ORDER
            ),
            "source_hashes": self.source_hashes(model_id, scenario_id),
            "artifacts": {
                path.name: sha256_file(path)
                for path in (prediction_path, calibrator_path, threshold_path)
            },
        }
        atomic_write_json(output / "candidate_selection.json", selection)
        print(
            f"[stress selected] {model_id}/{scenario_id}: "
            f"{selected['candidate_id']}",
            flush=True,
        )
        return selection

    def run_test(self, model_id: str, scenario_id: str) -> dict[str, Any]:
        scenario = self.scenarios.scenario(scenario_id)
        selection_path = selection_dir(model_id, scenario_id) / "candidate_selection.json"
        if not selection_path.is_file():
            raise FileNotFoundError(f"stress selection is incomplete: {selection_path}")
        selection = read_json(selection_path)
        if selection.get("status") != "complete" or selection.get("outer_test_accessed") is not False:
            raise RuntimeError("stress selection artifact is invalid")
        selection_sha = sha256_file(selection_path)
        candidate = dict(selection["selected_candidate"]["candidate"])
        candidate_id = str(candidate["candidate_id"])
        fixed_epochs = selection["selected_candidate"].get(
            "selected_final_epoch_if_chosen"
        )
        train_ids = scenario.train_ids
        test_ids = scenario.test_ids
        source_hashes = self.source_hashes(model_id, scenario_id)
        if model_id in CLASSICAL_MODEL_IDS:
            seeds = (
                int(self.engine.validation["model_selection"]["selection_seed"]),
            )
        else:
            seeds = tuple(
                int(value)
                for value in self.engine.modeling["neural_training"][
                    "final_outer_seeds"
                ]
            )
            if fixed_epochs is None or int(fixed_epochs) < 1:
                raise RuntimeError("selected neural stress model has no epoch estimate")
        matrices: list[np.ndarray] = []
        rows_by_seed: list[list[dict[str, Any]]] = []
        run_ids: list[str] = []
        for seed in seeds:
            training_config = (
                self.engine.neural_training_config(model_id, candidate, seed=seed)
                if model_id not in CLASSICAL_MODEL_IDS
                else None
            )
            configuration = {
                "scope": "registered_domain_shift_final_test",
                "scenario_id": scenario_id,
                "scenario_group": scenario_group(scenario_id),
                "model_id": model_id,
                "family": MODEL_FAMILIES[model_id],
                "candidate": candidate,
                "fixed_epochs": fixed_epochs,
                "training": None if training_config is None else training_config.__dict__,
                "seed": int(seed),
                "device": self.device,
                "selection_artifact_sha256": selection_sha,
                "outer_test_accessed": True,
                "forward_validation_not_in_final_training": bool(
                    scenario.uses_fixed_forward_validation
                ),
            }
            reusable = self.engine.completed_run(
                stage=TEST_STAGE,
                model_id=model_id,
                outer_fold_id=scenario_id,
                inner_fold_id=None,
                candidate_id=candidate_id,
                seed=seed,
                configuration=configuration,
                train_ids=train_ids,
                validation_ids=test_ids,
                source_hashes=source_hashes,
            )
            if reusable is not None:
                run_dir, _, prediction_rows = reusable
                if not self.quiet_trials:
                    print(f"[stress test resume] {run_dir.name}", flush=True)
            else:
                ledger = RunLedger.initialize(
                    RUN_ROOT,
                    stage=TEST_STAGE,
                    model_id=model_id,
                    outer_fold_id=scenario_id,
                    inner_fold_id=None,
                    candidate_id=candidate_id,
                    seed=seed,
                    configuration=configuration,
                    train_encounter_ids=train_ids,
                    validation_encounter_ids=test_ids,
                    source_hashes=source_hashes,
                )
                print(f"[stress test run] {ledger.run_id}", flush=True)
                if model_id in CLASSICAL_MODEL_IDS:
                    fit = fit_classical_baseline(
                        self.engine.registry,
                        train_ids,
                        test_ids,
                        model_id=model_id,
                        modeling_config=self.engine.modeling,
                        candidate=None if model_id == "prior_constant" else candidate,
                        random_seed=seed,
                    )
                    ledger.save_classical_result(
                        fit,
                        model_id=model_id,
                        outer_test_accessed=True,
                    )
                else:
                    assert training_config is not None
                    seed_everything(seed, deterministic_algorithms=True)
                    model = self.engine.build_model(model_id, candidate)
                    train_dataset, test_dataset, preprocessing = self.engine.build_datasets(
                        model_id,
                        candidate,
                        train_ids,
                        test_ids,
                        seed=seed,
                    )
                    dataset_preprocessing_record(train_dataset, test_dataset)
                    fit = fit_neural_fixed_epochs(
                        model,
                        train_dataset,
                        training_config,
                        epochs=int(fixed_epochs),
                        device=self.device,
                    )
                    prediction = predict_mil(
                        model,
                        test_dataset,
                        batch_size=training_config.encounter_batch_size,
                        device=torch.device(self.device),
                        data_loader_workers=0,
                    )
                    ledger.save_fixed_epoch_result(
                        fit,
                        prediction,
                        model_id=model_id,
                        preprocessing=preprocessing,
                        selection_artifact_sha256=selection_sha,
                    )
                    del model, train_dataset, test_dataset, fit, prediction
                    gc.collect()
                run_dir = ledger.run_dir
                prediction_rows = read_jsonl(run_dir / "predictions.jsonl")
            ordered = sorted(prediction_rows, key=lambda row: row["encounter_id"])
            if tuple(row["encounter_id"] for row in ordered) != tuple(sorted(test_ids)):
                raise RuntimeError("stress test checkpoint prediction coverage mismatch")
            matrices.append(
                np.asarray([row["probability"] for row in ordered], dtype=np.float64)
            )
            rows_by_seed.append(ordered)
            run_ids.append(run_dir.name)
        raw = ensemble_prediction_matrices(matrices)
        reference = rows_by_seed[0]
        truth = np.asarray([row["y_true"] for row in reference], dtype=np.int8)
        calibrator = read_json(selection_path.parent / "calibrator.json")
        calibrated = apply_calibrator(raw, calibrator)
        output = test_dir(model_id, scenario_id)
        output.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "model_id": model_id,
                "scenario_id": scenario_id,
                "scenario_group": scenario_group(scenario_id),
                "encounter_id": str(reference[index]["encounter_id"]),
                "campaign_id": str(reference[index]["campaign_id"]),
                "label_order": list(LABEL_ORDER),
                "y_true": truth[index].astype(int).tolist(),
                "seed_probabilities": [
                    matrix[index].astype(float).tolist() for matrix in matrices
                ],
                "raw_probability": raw[index].astype(float).tolist(),
                "calibrated_probability": calibrated[index].astype(float).tolist(),
                "label_scope": "encounter_bag",
            }
            for index in range(len(reference))
        ]
        prediction_path = output / "test_predictions.jsonl"
        atomic_write_jsonl(prediction_path, rows)
        result = {
            "status": "complete",
            "scope": "registered_domain_shift_final_test",
            "model_id": model_id,
            "scenario_id": scenario_id,
            "scenario_group": scenario_group(scenario_id),
            "train_count": len(train_ids),
            "fixed_validation_count": len(scenario.validation_ids),
            "test_count": len(test_ids),
            "buffer_excluded_count": len(scenario.buffer_ids),
            "train_partition_sha256": partition_sha256(train_ids),
            "test_partition_sha256": partition_sha256(test_ids),
            "selected_candidate_id": candidate_id,
            "fixed_epochs": fixed_epochs,
            "final_seeds": list(seeds),
            "run_ids": run_ids,
            "selection_artifact_sha256": selection_sha,
            "test_predictions_sha256": sha256_file(prediction_path),
            "raw_metrics": probability_metrics(truth, raw, LABEL_ORDER),
            "calibrated_metrics": probability_metrics(
                truth, calibrated, LABEL_ORDER
            ),
            "source_hashes": source_hashes,
            "interpretation_guard": self.execution["interpretation_guard"][
                scenario_group(scenario_id)
            ],
        }
        atomic_write_json(output / "test_result.json", result)
        print(
            f"[stress test complete] {model_id}/{scenario_id} raw AP="
            f"{result['raw_metrics']['macro_average_precision']:.6f}",
            flush=True,
        )
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("selection", "test", "all", "dry-run"), default="all")
    parser.add_argument("--models", default="all")
    parser.add_argument("--scenarios", default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--quiet-trials", action="store_true")
    args = parser.parse_args()
    execution_path = PROJECT_ROOT / "configs/stress_execution_v1.toml"
    if not execution_path.is_file():
        raise FileNotFoundError(
            "freeze configs/stress_execution_v1.toml before stress execution"
        )
    execution = load_toml(execution_path)
    scenario_registry = StressScenarioRegistry.load(PROJECT_ROOT)
    allowed_pairs = configured_pairs(execution, scenario_registry)
    allowed_models = tuple(model for model in MODEL_ORDER if any(pair[0] == model for pair in allowed_pairs))
    models = parse_subset(args.models, allowed_models, "models")
    scenarios = parse_subset(args.scenarios, scenario_registry.scenario_ids, "scenarios")
    pairs = tuple(
        pair for pair in allowed_pairs if pair[0] in models and pair[1] in scenarios
    )
    if not pairs:
        raise ValueError("requested model/scenario filters select no registered pair")
    if args.mode == "dry-run":
        print(
            json.dumps(
                {
                    "status": "ready",
                    "pair_count": len(pairs),
                    "pairs": [
                        {"model_id": model_id, "scenario_id": scenario_id}
                        for model_id, scenario_id in pairs
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    runner = RegisteredStressRunner(
        models=tuple(sorted({pair[0] for pair in pairs})),
        device=args.device,
        torch_threads=args.torch_threads,
        quiet_trials=args.quiet_trials,
    )
    # HF-LW-GAM is ordered before its matched mean ablation in MODEL_ORDER.
    for model_id, scenario_id in pairs:
        if args.mode in {"selection", "all"}:
            runner.run_selection(model_id, scenario_id)
        if args.mode in {"test", "all"}:
            runner.run_test(model_id, scenario_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
