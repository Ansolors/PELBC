#!/usr/bin/env python3
"""Resumable grouped nested-CV selection and outer-fold evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dolphin_behavior_mil.classical import fit_classical_baseline  # noqa: E402
from dolphin_behavior_mil.evaluation import evaluation_record  # noqa: E402
from dolphin_behavior_mil.experiment import (  # noqa: E402
    MODEL_FAMILIES,
    RunLedger,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_sha256,
    load_toml,
    run_identity,
    sha256_file,
)
from dolphin_behavior_mil.mil_models import (  # noqa: E402
    build_cnn_mil_model,
    build_frozen_embedding_mil_model,
)
from dolphin_behavior_mil.modeling_data import (  # noqa: E402
    DatasetRegistry,
    FeatureNormalizer,
    FrozenEmbeddingBagDataset,
    FrozenEmbeddingRegistry,
    LABEL_ORDER,
    MILBagDataset,
    NormalizedFeatureCache,
    partition_sha256,
)
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
from dolphin_behavior_mil.training import (  # noqa: E402
    NeuralTrainingConfig,
    dataset_preprocessing_record,
    fit_neural_fixed_epochs,
    fit_neural_model,
    predict_mil,
    seed_everything,
)


OUTER_FOLDS = tuple(f"outer_{index:02d}" for index in range(1, 6))
INNER_FOLDS = tuple(f"inner_{index:02d}" for index in range(1, 5))
MATCHED_ABLATION_IDS = (
    "cnn_mean",
    "cnn_max",
    "cnn_linear_softmax",
    "cnn_shared_gated_attention",
)
DEFAULT_MODEL_ORDER = (
    "prior_constant",
    "bag_size_logistic",
    "metadata_logistic",
    "handcrafted_logistic",
    "hf_lw_gam",
    "cnn_mean",
    "cnn_max",
    "cnn_linear_softmax",
    "cnn_shared_gated_attention",
    "panns_frozen_mil",
    "aves_frozen_mil",
)
RUN_ROOT = PROJECT_ROOT / "results/model_runs"
NESTED_ROOT = PROJECT_ROOT / "results/nested_cv"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selection_dir(model_id: str, outer_fold_id: str) -> Path:
    return NESTED_ROOT / "selection" / model_id / outer_fold_id


def outer_result_dir(model_id: str, outer_fold_id: str) -> Path:
    return NESTED_ROOT / "outer_folds" / model_id / outer_fold_id


def parse_csv_choice(value: str, allowed: Sequence[str]) -> tuple[str, ...]:
    if value == "all":
        return tuple(allowed)
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(parsed) - set(allowed))
    if not parsed or unknown:
        raise ValueError(f"invalid choices {unknown or parsed}; allowed={list(allowed)}")
    return parsed


def require_locked_selection(
    path: Path,
    *,
    model_id: str,
    outer_fold_id: str,
) -> dict[str, Any]:
    """Outer evaluation gate: no outer run may start without completed inner selection."""

    if not path.is_file():
        raise FileNotFoundError(f"inner selection is incomplete: {path}")
    record = read_json(path)
    if record.get("status") != "complete":
        raise RuntimeError("inner selection artifact is not complete")
    if record.get("model_id") != model_id or record.get("outer_fold_id") != outer_fold_id:
        raise RuntimeError("inner selection artifact identity mismatch")
    if record.get("outer_test_accessed") is not False:
        raise RuntimeError("inner selection artifact has an invalid outer-test access flag")
    if not record.get("selected_candidate", {}).get("candidate_id"):
        raise RuntimeError("inner selection artifact has no selected candidate")
    return record


class NestedRunner:
    def __init__(
        self,
        *,
        models: Sequence[str],
        device: str,
        torch_threads: int,
        preload_logmel: bool,
        quiet_trials: bool,
        encoder_microbatch_override: int | None,
    ) -> None:
        self.models = tuple(models)
        self.device = str(device)
        self.torch_threads = int(torch_threads)
        self.quiet_trials = bool(quiet_trials)
        self.encoder_microbatch_override = encoder_microbatch_override
        torch.set_num_threads(self.torch_threads)
        self.modeling = load_toml(PROJECT_ROOT / "configs/modeling_protocol_v1.toml")
        self.validation = load_toml(PROJECT_ROOT / "configs/validation_protocol_v1.toml")
        self.pretrained = load_toml(PROJECT_ROOT / "configs/pretrained_encoders_v1.toml")
        self.cnn_compute = load_toml(PROJECT_ROOT / "configs/cnn_compute_protocol_v1.toml")
        self.registry = DatasetRegistry.load(
            PROJECT_ROOT,
            cache_features_in_memory=bool(set(models) & CNN_MODEL_IDS),
        )
        self.embedding_registries: dict[str, FrozenEmbeddingRegistry] = {}
        self.normalizers: dict[str, FeatureNormalizer] = {}
        self.active_normalized_cache_key: str | None = None
        self.active_normalized_cache: NormalizedFeatureCache | None = None
        self.source_hash_cache: dict[str, str] = {}
        if preload_logmel and set(models) & CNN_MODEL_IDS:
            start = time.perf_counter()
            record = self.registry.preload_feature_values()
            record["elapsed_seconds"] = time.perf_counter() - start
            print(f"preloaded log-mel cache: {record}", flush=True)

    def source_hash(self, relative: str) -> str:
        path = PROJECT_ROOT / relative
        key = str(path)
        if key not in self.source_hash_cache:
            self.source_hash_cache[key] = sha256_file(path)
        return self.source_hash_cache[key]

    def source_hashes(
        self,
        model_id: str,
        *,
        outer_fold_id: str | None = None,
    ) -> dict[str, str]:
        relatives = {
            "modeling_protocol": "configs/modeling_protocol_v1.toml",
            "validation_protocol": "configs/validation_protocol_v1.toml",
            "dataset_manifest": "data/processed/v0.4.0/manifest.json",
            "outer_assignments": "data/processed/v0.4.0/outer_fold_assignments.jsonl",
            "inner_assignments": "data/processed/v0.4.0/inner_fold_assignments.jsonl",
        }
        if model_id in PRETRAINED_MODEL_IDS:
            relatives["pretrained_protocol"] = "configs/pretrained_encoders_v1.toml"
            key = "panns" if model_id == "panns_frozen_mil" else "aves"
            relatives["embedding_manifest"] = str(
                self.pretrained["encoders"][key]["manifest_path"]
            )
        if model_id in CNN_MODEL_IDS:
            relatives["cnn_compute_protocol"] = "configs/cnn_compute_protocol_v1.toml"
        hashes = {name: self.source_hash(path) for name, path in relatives.items()}
        if model_id in MATCHED_ABLATION_IDS and outer_fold_id is not None:
            dependency = selection_dir("hf_lw_gam", outer_fold_id) / "candidate_selection.json"
            require_locked_selection(
                dependency,
                model_id="hf_lw_gam",
                outer_fold_id=outer_fold_id,
            )
            hashes["matched_hf_lw_gam_selection"] = sha256_file(dependency)
        return hashes

    def candidates_for_outer(
        self,
        model_id: str,
        outer_fold_id: str,
    ) -> list[dict[str, Any]]:
        if model_id not in MATCHED_ABLATION_IDS:
            candidates = candidates_for_model(self.modeling, model_id)
            if model_id == "hf_lw_gam":
                eligible = set(
                    self.cnn_compute["candidate_subset"]["eligible_candidate_ids"]
                )
                candidates = [
                    candidate
                    for candidate in candidates
                    if candidate["candidate_id"] in eligible
                ]
                if {candidate["candidate_id"] for candidate in candidates} != eligible:
                    raise RuntimeError("CNN compute candidate subset does not match registry")
            return candidates
        dependency = require_locked_selection(
            selection_dir("hf_lw_gam", outer_fold_id) / "candidate_selection.json",
            model_id="hf_lw_gam",
            outer_fold_id=outer_fold_id,
        )
        return [dict(dependency["selected_candidate"]["candidate"])]

    def embedding_registry(self, model_id: str) -> FrozenEmbeddingRegistry:
        if model_id not in PRETRAINED_MODEL_IDS:
            raise KeyError(model_id)
        key = "panns" if model_id == "panns_frozen_mil" else "aves"
        if key not in self.embedding_registries:
            config = self.pretrained["encoders"][key]
            print(f"[{model_id}] validating and preloading frozen embeddings", flush=True)
            self.embedding_registries[key] = FrozenEmbeddingRegistry.load(
                PROJECT_ROOT,
                PROJECT_ROOT / str(config["manifest_path"]),
                self.registry,
                verify_files=True,
                preload_values=True,
            )
        return self.embedding_registries[key]

    def normalizer(self, train_ids: Sequence[str]) -> FeatureNormalizer:
        key = partition_sha256(train_ids)
        if key not in self.normalizers:
            self.normalizers[key] = FeatureNormalizer.fit(self.registry, train_ids)
        return self.normalizers[key]

    def normalized_cache(
        self,
        normalizer: FeatureNormalizer,
        encounter_ids: Sequence[str],
    ) -> NormalizedFeatureCache:
        key = normalizer.training_partition_sha256
        if self.active_normalized_cache_key != key:
            self.active_normalized_cache = None
            gc.collect()
            start = time.perf_counter()
            self.active_normalized_cache = NormalizedFeatureCache.build(
                self.registry, normalizer, encounter_ids
            )
            self.active_normalized_cache_key = key
            print(
                f"normalized fold cache: {len(self.active_normalized_cache.values_by_whistle)} "
                f"instances, {self.active_normalized_cache.bytes / 2**20:.1f} MiB, "
                f"{time.perf_counter() - start:.2f}s",
                flush=True,
            )
        assert self.active_normalized_cache is not None
        return self.active_normalized_cache

    def neural_training_config(
        self,
        model_id: str,
        candidate: Mapping[str, Any],
        *,
        seed: int,
        maximum_epochs_override: int | None = None,
    ) -> NeuralTrainingConfig:
        common = self.modeling["neural_training"]
        batch_size = (
            int(candidate["encounter_batch_size"])
            if "encounter_batch_size" in candidate
            else int(self.modeling["pretrained_frozen_mil"]["encounter_batch_size"])
        )
        if maximum_epochs_override is not None:
            maximum_epochs = int(maximum_epochs_override)
            patience = min(int(common["early_stopping_patience"]), maximum_epochs)
        elif model_id in CNN_MODEL_IDS:
            maximum_epochs = int(self.cnn_compute["training"]["maximum_epochs"])
            patience = int(self.cnn_compute["training"]["early_stopping_patience"])
        else:
            maximum_epochs = int(common["maximum_epochs"])
            patience = int(common["early_stopping_patience"])
        return NeuralTrainingConfig(
            learning_rate=float(candidate["learning_rate"]),
            weight_decay=float(candidate["weight_decay"]),
            encounter_batch_size=batch_size,
            maximum_epochs=maximum_epochs,
            early_stopping_patience=min(patience, maximum_epochs),
            early_stopping_minimum_delta=float(
                common["early_stopping_minimum_delta"]
            ),
            gradient_clip_norm=float(common["gradient_clip_norm"]),
            seed=int(seed),
            deterministic_algorithms=bool(common["deterministic_algorithms"]),
            data_loader_workers=0,
        )

    def build_model(self, model_id: str, candidate: Mapping[str, Any]) -> Any:
        if model_id in CNN_MODEL_IDS:
            return build_cnn_mil_model(
                model_id,
                base_channels=int(candidate["base_channels"]),
                embedding_dim=int(candidate["embedding_dim"]),
                attention_dim=int(candidate["attention_dim"]),
                dropout=float(candidate["dropout"]),
                encoder_microbatch_max_instances=int(
                    self.encoder_microbatch_override
                    if self.encoder_microbatch_override is not None
                    else self.modeling["neural_training"][
                        "encoder_microbatch_max_instances"
                    ]
                ),
            )
        embedding_registry = self.embedding_registry(model_id)
        return build_frozen_embedding_mil_model(
            model_id,
            embedding_dim=embedding_registry.embedding_dimension,
            attention_dim=int(candidate["attention_dim"]),
            dropout=float(candidate["dropout"]),
        )

    def build_datasets(
        self,
        model_id: str,
        candidate: Mapping[str, Any],
        train_ids: Sequence[str],
        validation_ids: Sequence[str],
        *,
        seed: int,
    ) -> tuple[Any, Any, dict[str, Any]]:
        if model_id in CNN_MODEL_IDS:
            normalizer = self.normalizer(train_ids)
            normalized_cache = self.normalized_cache(
                normalizer, tuple(sorted(set(train_ids) | set(validation_ids)))
            )
            augmentation = self.modeling["augmentation"][str(candidate["augmentation"])]
            train_dataset = MILBagDataset(
                self.registry,
                train_ids,
                normalizer,
                training=True,
                maximum_instances_per_bag=int(
                    candidate["maximum_training_instances_per_bag"]
                ),
                seed=seed,
                augmentation=augmentation,
                normalized_feature_cache=normalized_cache,
            )
            validation_dataset = MILBagDataset(
                self.registry,
                validation_ids,
                normalizer,
                training=False,
                normalized_feature_cache=normalized_cache,
            )
        else:
            embedding_registry = self.embedding_registry(model_id)
            train_dataset = FrozenEmbeddingBagDataset(
                self.registry,
                embedding_registry,
                train_ids,
                training=True,
                maximum_instances_per_bag=None,
                seed=seed,
            )
            validation_dataset = FrozenEmbeddingBagDataset(
                self.registry,
                embedding_registry,
                validation_ids,
                training=False,
            )
        preprocessing = dataset_preprocessing_record(train_dataset, validation_dataset)
        embedding_manifest = preprocessing.get("embedding_manifest")
        if embedding_manifest:
            preprocessing["embedding_manifest"] = str(
                Path(embedding_manifest).resolve().relative_to(PROJECT_ROOT)
            )
        return train_dataset, validation_dataset, preprocessing

    def completed_run(
        self,
        *,
        stage: str,
        model_id: str,
        outer_fold_id: str,
        inner_fold_id: str | None,
        candidate_id: str,
        seed: int,
        configuration: Mapping[str, Any],
        train_ids: Sequence[str],
        validation_ids: Sequence[str],
        source_hashes: Mapping[str, str],
    ) -> tuple[Path, dict[str, Any], list[dict[str, Any]]] | None:
        run_id, identity = run_identity(
            stage=stage,
            model_id=model_id,
            outer_fold_id=outer_fold_id,
            inner_fold_id=inner_fold_id,
            candidate_id=candidate_id,
            seed=seed,
        )
        run_dir = RUN_ROOT / run_id
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            return None
        manifest = read_json(manifest_path)
        if manifest.get("status") != "completed":
            return None
        expected = {
            "identity": identity,
            "configuration_sha256": canonical_sha256(configuration),
            "train_partition_sha256": partition_sha256(train_ids),
            "validation_partition_sha256": partition_sha256(validation_ids),
            "source_hashes": dict(source_hashes),
        }
        mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
        if mismatches:
            raise RuntimeError(
                f"completed run identity collision for {run_id}: {mismatches}"
            )
        for name, expected_hash in manifest.get("artifact_sha256", {}).items():
            if sha256_file(run_dir / name) != expected_hash:
                raise RuntimeError(f"completed run artifact hash mismatch: {run_id}/{name}")
        return run_dir, read_json(run_dir / "metrics.json"), read_jsonl(
            run_dir / "predictions.jsonl"
        )

    def trial_record(
        self,
        run_dir: Path,
        metrics: Mapping[str, Any],
        predictions: Sequence[Mapping[str, Any]],
        *,
        inner_fold_id: str,
    ) -> dict[str, Any]:
        score = float(metrics["best_validation_macro_average_precision"])
        return {
            "run_id": run_dir.name,
            "run_directory_relative_to_project": str(run_dir.relative_to(PROJECT_ROOT)),
            "inner_fold_id": inner_fold_id,
            "macro_average_precision": score,
            "best_epoch": metrics.get("best_epoch"),
            "trainable_parameters": int(metrics.get("trainable_parameters") or 0),
            "median_encounter_inference_ms": float(
                metrics.get("median_encounter_inference_ms") or 0.0
            ),
            "prediction_count": len(predictions),
        }

    def run_inner_trial(
        self,
        *,
        model_id: str,
        outer_fold_id: str,
        inner_fold_id: str,
        candidate: Mapping[str, Any],
        stage: str = "inner_selection",
        maximum_epochs_override: int | None = None,
    ) -> dict[str, Any]:
        train_ids, validation_ids = self.registry.inner_split(
            outer_fold_id, inner_fold_id
        )
        seed = int(self.validation["model_selection"]["selection_seed"])
        candidate_id = str(candidate["candidate_id"])
        source_hashes = self.source_hashes(
            model_id,
            outer_fold_id=(
                outer_fold_id if stage.startswith("inner_selection") else None
            ),
        )
        training_config = (
            self.neural_training_config(
                model_id,
                candidate,
                seed=seed,
                maximum_epochs_override=maximum_epochs_override,
            )
            if model_id not in CLASSICAL_MODEL_IDS
            else None
        )
        configuration = {
            "scope": (
                "official_inner_model_selection"
                if stage.startswith("inner_selection")
                else "scheduler_resource_smoke_not_scientific_result"
            ),
            "model_id": model_id,
            "family": MODEL_FAMILIES[model_id],
            "outer_fold_id": outer_fold_id,
            "inner_fold_id": inner_fold_id,
            "candidate": dict(candidate),
            "training": None if training_config is None else training_config.__dict__,
            "device": self.device,
            "encoder_microbatch_max_instances": (
                self.encoder_microbatch_override
                if self.encoder_microbatch_override is not None
                else self.modeling["neural_training"][
                    "encoder_microbatch_max_instances"
                ]
                if model_id in CNN_MODEL_IDS
                else None
            ),
            "outer_test_accessed": False,
        }
        reusable = self.completed_run(
            stage=stage,
            model_id=model_id,
            outer_fold_id=outer_fold_id,
            inner_fold_id=inner_fold_id,
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
                print(f"[resume] {run_dir.name}", flush=True)
            return self.trial_record(
                run_dir, metrics, predictions, inner_fold_id=inner_fold_id
            )

        ledger = RunLedger.initialize(
            RUN_ROOT,
            stage=stage,
            model_id=model_id,
            outer_fold_id=outer_fold_id,
            inner_fold_id=inner_fold_id,
            candidate_id=candidate_id,
            seed=seed,
            configuration=configuration,
            train_encounter_ids=train_ids,
            validation_encounter_ids=validation_ids,
            source_hashes=source_hashes,
        )
        if not self.quiet_trials:
            print(f"[run] {ledger.run_id}", flush=True)
        if model_id in CLASSICAL_MODEL_IDS:
            fit = fit_classical_baseline(
                self.registry,
                train_ids,
                validation_ids,
                model_id=model_id,
                modeling_config=self.modeling,
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
            model = self.build_model(model_id, candidate)
            train_dataset, validation_dataset, preprocessing = self.build_datasets(
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
            if self.device == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
        metrics = read_json(ledger.run_dir / "metrics.json")
        predictions = read_jsonl(ledger.run_dir / "predictions.jsonl")
        return self.trial_record(
            ledger.run_dir, metrics, predictions, inner_fold_id=inner_fold_id
        )

    def run_selection(self, model_id: str, outer_fold_id: str) -> dict[str, Any]:
        candidates = self.candidates_for_outer(model_id, outer_fold_id)
        inner_stage = (
            f"inner_selection_{self.cnn_compute['protocol']['run_stage_suffix']}"
            if model_id in CNN_MODEL_IDS
            else "inner_selection"
        )
        summaries = []
        records_by_candidate = {
            str(candidate["candidate_id"]): [] for candidate in candidates
        }
        for inner_fold_id in INNER_FOLDS:
            for candidate in candidates:
                candidate_id = str(candidate["candidate_id"])
                records_by_candidate[candidate_id].append(
                    self.run_inner_trial(
                        model_id=model_id,
                        outer_fold_id=outer_fold_id,
                        inner_fold_id=inner_fold_id,
                        candidate=candidate,
                        stage=inner_stage,
                    )
                )
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            fold_records = records_by_candidate[candidate_id]
            summaries.append(
                summarize_candidate(
                    candidate,
                    fold_records,
                    expected_fold_count=len(INNER_FOLDS),
                )
            )
            print(
                f"[{model_id}/{outer_fold_id}] {candidate_id} "
                f"mean inner AP={summaries[-1]['mean_inner_validation_macro_average_precision']:.6f}",
                flush=True,
            )
        selected = select_candidate(
            summaries,
            tie_tolerance_macro_ap=float(
                self.validation["model_selection"]["tie_tolerance_macro_ap"]
            ),
        )
        outer_train_ids, _ = self.registry.outer_split(outer_fold_id)
        selected_run_records = records_by_candidate[str(selected["candidate_id"])]
        selected_prediction_rows = []
        for record in selected_run_records:
            selected_prediction_rows.extend(
                read_jsonl(
                    PROJECT_ROOT
                    / str(record["run_directory_relative_to_project"])
                    / "predictions.jsonl"
                )
            )
        oof = assemble_oof_rows(selected_prediction_rows, outer_train_ids)
        for index, encounter_id in enumerate(oof["encounter_ids"]):
            if not np.array_equal(
                oof["labels"][index], self.registry.labels[encounter_id].astype(np.int8)
            ):
                raise RuntimeError(f"OOF label mismatch for {encounter_id}")
            if oof["campaign_ids"][index] != self.registry.campaign_by_encounter[encounter_id]:
                raise RuntimeError(f"OOF campaign mismatch for {encounter_id}")

        calibration_config = self.validation["calibration"]
        if model_id == "prior_constant" and not bool(
            calibration_config["constant_prior_baseline_recalibration"]
        ):
            calibrator = identity_calibrator()
        else:
            calibrator = fit_platt_calibrator(
                oof["labels"],
                oof["probabilities"],
                c_value=float(calibration_config["l2_inverse_regularization_c"]),
                probability_clip=calibration_config["probability_clip"],
                random_seed=int(self.validation["model_selection"]["selection_seed"]),
            )
        calibrated = apply_calibrator(oof["probabilities"], calibrator)
        thresholds = select_multilabel_thresholds(oof["labels"], calibrated)
        output_dir = selection_dir(model_id, outer_fold_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        oof_rows = []
        for index, encounter_id in enumerate(oof["encounter_ids"]):
            oof_rows.append(
                {
                    "model_id": model_id,
                    "outer_fold_id": outer_fold_id,
                    "encounter_id": encounter_id,
                    "campaign_id": oof["campaign_ids"][index],
                    "label_order": list(LABEL_ORDER),
                    "y_true": oof["labels"][index].astype(int).tolist(),
                    "raw_probability": oof["probabilities"][index].astype(float).tolist(),
                    "calibrated_probability": calibrated[index].astype(float).tolist(),
                    "outer_test_accessed": False,
                }
            )
        atomic_write_jsonl(output_dir / "selected_inner_oof_predictions.jsonl", oof_rows)
        atomic_write_json(output_dir / "calibrator.json", calibrator)
        atomic_write_json(output_dir / "thresholds.json", thresholds)
        selection = {
            "status": "complete",
            "scope": "official_inner_model_selection",
            "model_id": model_id,
            "outer_fold_id": outer_fold_id,
            "outer_train_encounter_count": len(outer_train_ids),
            "outer_train_partition_sha256": partition_sha256(outer_train_ids),
            "outer_test_accessed": False,
            "candidate_count": len(candidates),
            "inner_fold_count": len(INNER_FOLDS),
            "selection_metric": self.validation["model_selection"]["selection_metric"],
            "selection_strategy": (
                "matched_hf_lw_gam_candidate_no_independent_reoptimization"
                if model_id in MATCHED_ABLATION_IDS
                else "independent_prespecified_inner_grid"
            ),
            "candidate_summaries": summaries,
            "selected_candidate": selected,
            "selected_inner_oof_raw_macro_average_precision": float(
                oof["macro_average_precision"]
            ),
            "selected_inner_oof_calibrated_metrics": evaluation_record(
                oof["labels"], calibrated, LABEL_ORDER
            ),
            "source_hashes": self.source_hashes(
                model_id, outer_fold_id=outer_fold_id
            ),
            "artifacts": {
                name: sha256_file(output_dir / name)
                for name in (
                    "selected_inner_oof_predictions.jsonl",
                    "calibrator.json",
                    "thresholds.json",
                )
            },
        }
        atomic_write_json(output_dir / "candidate_selection.json", selection)
        print(
            f"[selected] {model_id}/{outer_fold_id}: {selected['candidate_id']}",
            flush=True,
        )
        return selection

    def _final_configuration(
        self,
        *,
        model_id: str,
        outer_fold_id: str,
        candidate: Mapping[str, Any],
        seed: int,
        fixed_epochs: int | None,
        selection_sha256: str,
        training_config: NeuralTrainingConfig | None,
    ) -> dict[str, Any]:
        return {
            "scope": "official_outer_test_evaluation_after_locked_inner_selection",
            "model_id": model_id,
            "family": MODEL_FAMILIES[model_id],
            "outer_fold_id": outer_fold_id,
            "candidate": dict(candidate),
            "fixed_epochs": fixed_epochs,
            "training": None if training_config is None else training_config.__dict__,
            "seed": int(seed),
            "device": self.device,
            "selection_artifact_sha256": selection_sha256,
            "outer_test_accessed": True,
        }

    def run_final_outer(self, model_id: str, outer_fold_id: str) -> dict[str, Any]:
        selection_path = selection_dir(model_id, outer_fold_id) / "candidate_selection.json"
        selection = require_locked_selection(
            selection_path,
            model_id=model_id,
            outer_fold_id=outer_fold_id,
        )
        selection_sha = sha256_file(selection_path)
        selected = selection["selected_candidate"]
        candidate = dict(selected["candidate"])
        candidate_id = str(candidate["candidate_id"])
        fixed_epochs = selected.get("selected_final_epoch_if_chosen")
        outer_train_ids, outer_test_ids = self.registry.outer_split(outer_fold_id)
        if partition_sha256(outer_train_ids) != selection["outer_train_partition_sha256"]:
            raise RuntimeError("outer training pool changed after candidate selection")
        source_hashes = self.source_hashes(
            model_id, outer_fold_id=outer_fold_id
        )
        raw_matrices = []
        prediction_rows_by_seed = []
        if model_id in CLASSICAL_MODEL_IDS:
            seeds = (int(self.validation["model_selection"]["selection_seed"]),)
        else:
            seeds = tuple(int(value) for value in self.modeling["neural_training"]["final_outer_seeds"])
            if fixed_epochs is None or int(fixed_epochs) < 1:
                raise RuntimeError("neural selection has no valid final epoch")

        outer_normalizer = (
            self.normalizer(outer_train_ids) if model_id in CNN_MODEL_IDS else None
        )
        for seed in seeds:
            training_config = (
                self.neural_training_config(model_id, candidate, seed=seed)
                if model_id not in CLASSICAL_MODEL_IDS
                else None
            )
            configuration = self._final_configuration(
                model_id=model_id,
                outer_fold_id=outer_fold_id,
                candidate=candidate,
                seed=seed,
                fixed_epochs=None if fixed_epochs is None else int(fixed_epochs),
                selection_sha256=selection_sha,
                training_config=training_config,
            )
            outer_stage = (
                f"outer_final_{self.cnn_compute['protocol']['run_stage_suffix']}"
                if model_id in CNN_MODEL_IDS
                else "outer_final"
            )
            reusable = self.completed_run(
                stage=outer_stage,
                model_id=model_id,
                outer_fold_id=outer_fold_id,
                inner_fold_id=None,
                candidate_id=candidate_id,
                seed=seed,
                configuration=configuration,
                train_ids=outer_train_ids,
                validation_ids=outer_test_ids,
                source_hashes=source_hashes,
            )
            if reusable is not None:
                run_dir, _, prediction_rows = reusable
                print(f"[resume outer] {run_dir.name}", flush=True)
            else:
                ledger = RunLedger.initialize(
                    RUN_ROOT,
                    stage=outer_stage,
                    model_id=model_id,
                    outer_fold_id=outer_fold_id,
                    inner_fold_id=None,
                    candidate_id=candidate_id,
                    seed=seed,
                    configuration=configuration,
                    train_encounter_ids=outer_train_ids,
                    validation_encounter_ids=outer_test_ids,
                    source_hashes=source_hashes,
                )
                print(f"[outer run] {ledger.run_id}", flush=True)
                if model_id in CLASSICAL_MODEL_IDS:
                    fit = fit_classical_baseline(
                        self.registry,
                        outer_train_ids,
                        outer_test_ids,
                        model_id=model_id,
                        modeling_config=self.modeling,
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
                    model = self.build_model(model_id, candidate)
                    train_dataset, test_dataset, preprocessing = self.build_datasets(
                        model_id,
                        candidate,
                        outer_train_ids,
                        outer_test_ids,
                        seed=seed,
                    )
                    if outer_normalizer is not None and train_dataset.normalizer is not outer_normalizer:
                        raise RuntimeError("outer normalizer cache identity mismatch")
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
                    if self.device == "mps" and hasattr(torch, "mps"):
                        torch.mps.empty_cache()
                run_dir = ledger.run_dir
                prediction_rows = read_jsonl(run_dir / "predictions.jsonl")
            ordered = sorted(prediction_rows, key=lambda row: str(row["encounter_id"]))
            if tuple(str(row["encounter_id"]) for row in ordered) != tuple(
                sorted(outer_test_ids)
            ):
                raise RuntimeError("outer seed predictions do not exactly cover test fold")
            raw_matrices.append(
                np.asarray([row["probability"] for row in ordered], dtype=np.float64)
            )
            prediction_rows_by_seed.append(ordered)

        raw_probability = ensemble_prediction_matrices(raw_matrices)
        reference = prediction_rows_by_seed[0]
        truth = np.asarray([row["y_true"] for row in reference], dtype=np.int8)
        encounter_ids = tuple(str(row["encounter_id"]) for row in reference)
        campaign_ids = tuple(str(row["campaign_id"]) for row in reference)
        for rows in prediction_rows_by_seed[1:]:
            if [row["y_true"] for row in rows] != [row["y_true"] for row in reference]:
                raise RuntimeError("outer seed label ordering mismatch")
        calibrator = read_json(selection_dir(model_id, outer_fold_id) / "calibrator.json")
        thresholds = read_json(selection_dir(model_id, outer_fold_id) / "thresholds.json")
        calibrated_probability = apply_calibrator(raw_probability, calibrator)
        output_dir = outer_result_dir(model_id, outer_fold_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, encounter_id in enumerate(encounter_ids):
            rows.append(
                {
                    "model_id": model_id,
                    "outer_fold_id": outer_fold_id,
                    "encounter_id": encounter_id,
                    "campaign_id": campaign_ids[index],
                    "label_order": list(LABEL_ORDER),
                    "y_true": truth[index].astype(int).tolist(),
                    "seed_probabilities": [
                        matrix[index].astype(float).tolist() for matrix in raw_matrices
                    ],
                    "raw_probability": raw_probability[index].astype(float).tolist(),
                    "calibrated_probability": calibrated_probability[index]
                    .astype(float)
                    .tolist(),
                }
            )
        atomic_write_jsonl(output_dir / "outer_predictions.jsonl", rows)
        result = {
            "status": "complete",
            "model_id": model_id,
            "outer_fold_id": outer_fold_id,
            "outer_test_accessed": True,
            "selected_candidate_id": candidate_id,
            "fixed_epochs": fixed_epochs,
            "final_seeds": list(seeds),
            "outer_train_count": len(outer_train_ids),
            "outer_test_count": len(outer_test_ids),
            "outer_test_partition_sha256": partition_sha256(outer_test_ids),
            "raw_metrics": evaluation_record(truth, raw_probability, LABEL_ORDER),
            "calibrated_metrics": evaluation_record(
                truth, calibrated_probability, LABEL_ORDER
            ),
            "thresholds_fitted_from_inner_oof": thresholds,
            "selection_artifact_sha256": selection_sha,
            "outer_predictions_sha256": sha256_file(
                output_dir / "outer_predictions.jsonl"
            ),
        }
        atomic_write_json(output_dir / "outer_fold_result.json", result)
        print(
            f"[outer complete] {model_id}/{outer_fold_id} raw AP="
            f"{result['raw_metrics']['macro_average_precision']:.6f}",
            flush=True,
        )
        return result

    def aggregate_model_oof(self, model_id: str) -> dict[str, Any]:
        rows = []
        for outer_fold_id in OUTER_FOLDS:
            path = outer_result_dir(model_id, outer_fold_id) / "outer_predictions.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.extend(read_jsonl(path))
        by_id = {str(row["encounter_id"]): row for row in rows}
        if len(rows) != len(by_id) or set(by_id) != set(self.registry.encounter_ids):
            raise RuntimeError("outer OOF rows do not cover each encounter exactly once")
        ordered = [by_id[value] for value in self.registry.encounter_ids]
        truth = np.asarray([row["y_true"] for row in ordered], dtype=np.int8)
        raw = np.asarray([row["raw_probability"] for row in ordered], dtype=np.float64)
        calibrated = np.asarray(
            [row["calibrated_probability"] for row in ordered], dtype=np.float64
        )
        output_dir = NESTED_ROOT / "oof" / model_id
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_jsonl(output_dir / "oof_predictions.jsonl", ordered)
        result = {
            "status": "complete",
            "model_id": model_id,
            "encounter_count": len(ordered),
            "outer_fold_count": len(OUTER_FOLDS),
            "raw_metrics": evaluation_record(truth, raw, LABEL_ORDER),
            "calibrated_metrics": evaluation_record(truth, calibrated, LABEL_ORDER),
            "oof_predictions_sha256": sha256_file(
                output_dir / "oof_predictions.jsonl"
            ),
        }
        atomic_write_json(output_dir / "oof_metrics.json", result)
        return result


def schedule_summary(models: Sequence[str], outer_folds: Sequence[str]) -> dict[str, Any]:
    modeling = load_toml(PROJECT_ROOT / "configs/modeling_protocol_v1.toml")
    cnn_compute = load_toml(PROJECT_ROOT / "configs/cnn_compute_protocol_v1.toml")
    rows = []
    for model_id in models:
        if model_id in MATCHED_ABLATION_IDS:
            candidate_count = 1
        elif model_id == "hf_lw_gam":
            candidate_count = int(cnn_compute["candidate_subset"]["eligible_candidate_count"])
        else:
            candidate_count = len(candidates_for_model(modeling, model_id))
        inner_trials = candidate_count * len(INNER_FOLDS) * len(outer_folds)
        final_seed_count = 1 if model_id in CLASSICAL_MODEL_IDS else 3
        rows.append(
            {
                "model_id": model_id,
                "candidate_count_per_outer_fold": candidate_count,
                "inner_fold_count": len(INNER_FOLDS),
                "outer_fold_count": len(outer_folds),
                "inner_trial_count": inner_trials,
                "final_fit_count": final_seed_count * len(outer_folds),
            }
        )
    return {
        "models": rows,
        "total_inner_trials": sum(row["inner_trial_count"] for row in rows),
        "total_final_fits": sum(row["final_fit_count"] for row in rows),
        "outer_test_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("dry-run", "smoke", "inner", "outer", "all"), required=True
    )
    parser.add_argument("--models", default="all")
    parser.add_argument("--outer-folds", default="all")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--preload-logmel", action="store_true")
    parser.add_argument("--quiet-trials", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--smoke-encoder-microbatch", type=int)
    args = parser.parse_args()
    models = parse_csv_choice(args.models, DEFAULT_MODEL_ORDER)
    outer_folds = parse_csv_choice(args.outer_folds, OUTER_FOLDS)
    if args.torch_threads < 1 or args.smoke_epochs < 1:
        raise SystemExit("thread and smoke epoch counts must be positive")
    if args.smoke_encoder_microbatch is not None and (
        args.mode != "smoke" or args.smoke_encoder_microbatch < 1
    ):
        raise SystemExit("--smoke-encoder-microbatch is positive and smoke-only")
    if args.device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise SystemExit("MPS is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.mode == "dry-run":
        print(json.dumps(schedule_summary(models, outer_folds), ensure_ascii=False, indent=2))
        return 0

    runner = NestedRunner(
        models=models,
        device=args.device,
        torch_threads=args.torch_threads,
        preload_logmel=args.preload_logmel,
        quiet_trials=args.quiet_trials,
        encoder_microbatch_override=args.smoke_encoder_microbatch,
    )
    if args.mode == "smoke":
        records = []
        for model_id in models:
            candidate = candidates_for_model(runner.modeling, model_id)[0]
            records.append(
                runner.run_inner_trial(
                    model_id=model_id,
                    outer_fold_id=outer_folds[0],
                    inner_fold_id="inner_01",
                    candidate=candidate,
                    stage=(
                        f"scheduler_smoke_{args.device}_t{args.torch_threads}"
                        f"_m{args.smoke_encoder_microbatch or 32}"
                    ),
                    maximum_epochs_override=(
                        args.smoke_epochs if model_id not in CLASSICAL_MODEL_IDS else None
                    ),
                )
            )
        output = {
            "status": "complete",
            "scope": "scheduler_resource_smoke_not_scientific_result",
            "outer_test_accessed": False,
            "records": records,
        }
        atomic_write_json(NESTED_ROOT / "scheduler_smoke_summary.json", output)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0


    for model_id in models:
        if args.mode in {"inner", "all"}:
            for outer_fold_id in outer_folds:
                runner.run_selection(model_id, outer_fold_id)
        if args.mode in {"outer", "all"}:
            for outer_fold_id in outer_folds:
                runner.run_final_outer(model_id, outer_fold_id)
            if set(outer_folds) == set(OUTER_FOLDS):
                runner.aggregate_model_oof(model_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
