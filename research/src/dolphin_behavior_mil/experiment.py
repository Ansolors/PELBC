"""Auditable model registry and atomic run artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from .modeling_data import LABEL_ORDER, partition_sha256
from .evaluation import evaluation_record
from .training import FitResult, FixedEpochFitResult, PredictionBundle


MODEL_FAMILIES = {
    "prior_constant": "null",
    "bag_size_logistic": "nuisance_tabular",
    "metadata_logistic": "nuisance_tabular",
    "handcrafted_logistic": "traditional_acoustic",
    "cnn_mean": "learned_acoustic",
    "cnn_max": "learned_acoustic",
    "cnn_linear_softmax": "learned_acoustic",
    "cnn_shared_gated_attention": "learned_acoustic",
    "hf_lw_gam": "learned_acoustic_main",
    "panns_frozen_mil": "pretrained_acoustic",
    "aves_frozen_mil": "pretrained_bioacoustic",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_toml(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def _atomic_text(path: Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, target)


def atomic_write_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    _atomic_text(path, text)


def atomic_torch_save(path: Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="wb",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_modeling_protocol(
    modeling: Mapping[str, Any],
    method_scope: Mapping[str, Any],
    validation: Mapping[str, Any],
    audio: Mapping[str, Any],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    checks["dataset_version"] = (
        modeling["protocol"]["source_dataset_version"]
        == validation["protocol"]["output_dataset_version"]
    )
    checks["split_protocol"] = (
        modeling["protocol"]["split_protocol_id"]
        == validation["protocol"]["protocol_id"]
    )
    checks["preprocessing_id"] = (
        modeling["protocol"]["preprocessing_id"]
        == audio["preprocessing"]["preprocessing_id"]
    )
    checks["label_order"] = tuple(modeling["protocol"]["label_order"]) == LABEL_ORDER
    expected_models = [row["id"] for row in method_scope["required_models"]]
    registered_models = list(modeling["model_registry"]["required_model_ids"])
    checks["required_models"] = expected_models == registered_models
    checks["model_family_coverage"] = set(registered_models) == set(MODEL_FAMILIES)
    candidates = list(modeling["neural_candidates"])
    checks["neural_candidate_budget"] = (
        len(candidates)
        == int(validation["model_selection"]["neural_candidate_budget_per_outer_fold"])
    )
    checks["unique_neural_candidates"] = len({row["candidate_id"] for row in candidates}) == len(
        candidates
    )
    classical_count = len(modeling["classical"]["search"]["c_values"]) * len(
        modeling["classical"]["search"]["class_weight_options"]
    )
    checks["classical_candidate_budget"] = classical_count <= int(
        validation["model_selection"]["classical_candidate_budget_per_outer_fold"]
    )
    checks["main_counts"] = (
        int(modeling["data"]["primary_encounters"])
        == int(audio["instance_cohorts"]["main_encounters"])
        and int(modeling["data"]["main_instances"])
        == int(audio["instance_cohorts"]["main_whistles"])
    )
    checks["no_crop_or_resize"] = (
        modeling["data"]["variable_duration_policy"] == "no_crop_no_resize"
        and audio["duration"]["crop"] == "none"
    )
    checks["outer_test_guard"] = not bool(
        modeling["protocol"]["outer_test_access_during_selection"]
    ) and not bool(validation["outer_cv"]["outer_test_used_for_selection"])
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"modeling protocol cross-contract checks failed: {failed}")
    return {"checks": checks, "passed": len(checks), "failed": 0}


def build_model_registry(modeling: Mapping[str, Any]) -> list[dict[str, Any]]:
    required = list(modeling["model_registry"]["required_model_ids"])
    confirmatory = set(modeling["model_registry"]["confirmatory_model_ids"])
    records = []
    for index, model_id in enumerate(required, start=1):
        if model_id not in MODEL_FAMILIES:
            raise ValueError(f"unregistered model family for {model_id}")
        if model_id in {"prior_constant"}:
            input_kind = "training_label_prevalence"
            implementation = "PriorConstantBaseline"
        elif model_id in {"bag_size_logistic", "metadata_logistic", "handcrafted_logistic"}:
            input_kind = "fold_local_tabular"
            implementation = "IndependentLogisticBaseline"
        elif model_id in {"panns_frozen_mil", "aves_frozen_mil"}:
            input_kind = "frozen_embedding_bag"
            implementation = "IdentityEmbeddingEncoder+MILClassifier"
        else:
            input_kind = "variable_length_logmel_bag"
            implementation = "SmallWhistleCNN+MILClassifier"
        records.append(
            {
                "registry_order": index,
                "model_id": model_id,
                "family": MODEL_FAMILIES[model_id],
                "input_kind": input_kind,
                "implementation": implementation,
                "confirmatory_contrast_member": model_id in confirmatory,
                "selection_status": "prespecified_not_yet_selected",
            }
        )
    return records


def run_identity(
    *,
    stage: str,
    model_id: str,
    outer_fold_id: str,
    inner_fold_id: str | None,
    candidate_id: str,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    identity = {
        "stage": str(stage),
        "model_id": str(model_id),
        "outer_fold_id": str(outer_fold_id),
        "inner_fold_id": None if inner_fold_id is None else str(inner_fold_id),
        "candidate_id": str(candidate_id),
        "seed": int(seed),
    }
    digest = canonical_sha256(identity)[:12]
    run_id = "__".join(
        [
            identity["stage"],
            identity["model_id"],
            identity["outer_fold_id"],
            identity["inner_fold_id"] or "no_inner",
            identity["candidate_id"],
            str(identity["seed"]),
            digest,
        ]
    )
    return run_id, identity


@dataclass(frozen=True)
class RunLedger:
    run_id: str
    run_dir: Path
    identity: Mapping[str, Any]

    @classmethod
    def initialize(
        cls,
        root: Path,
        *,
        stage: str,
        model_id: str,
        outer_fold_id: str,
        inner_fold_id: str | None,
        candidate_id: str,
        seed: int,
        configuration: Mapping[str, Any],
        train_encounter_ids: Sequence[str],
        validation_encounter_ids: Sequence[str],
        source_hashes: Mapping[str, str],
    ) -> "RunLedger":
        run_id, identity = run_identity(
            stage=stage,
            model_id=model_id,
            outer_fold_id=outer_fold_id,
            inner_fold_id=inner_fold_id,
            candidate_id=candidate_id,
            seed=seed,
        )
        run_dir = Path(root) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        configuration_record = dict(configuration)
        atomic_write_json(run_dir / "configuration.json", configuration_record)
        manifest = {
            "run_id": run_id,
            "identity": identity,
            "status": "initialized",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration_sha256": canonical_sha256(configuration_record),
            "train_encounter_count": len(train_encounter_ids),
            "validation_encounter_count": len(validation_encounter_ids),
            "train_partition_sha256": partition_sha256(train_encounter_ids),
            "validation_partition_sha256": partition_sha256(validation_encounter_ids),
            "source_hashes": dict(source_hashes),
            "outer_test_accessed": False,
            "label_scope": "encounter_bag",
        }
        atomic_write_json(run_dir / "run_manifest.json", manifest)
        return cls(run_id=run_id, run_dir=run_dir, identity=identity)

    def save_fit_result(
        self,
        fit: FitResult,
        *,
        model_id: str,
        normalizer: Mapping[str, Any],
        label_order: Sequence[str] = LABEL_ORDER,
    ) -> None:
        atomic_write_json(self.run_dir / "normalizer.json", dict(normalizer))
        atomic_write_jsonl(self.run_dir / "history.jsonl", fit.history)
        checkpoint = {
            "model_id": str(model_id),
            "run_id": self.run_id,
            "best_epoch": fit.best_epoch,
            "state_dict": fit.best_state_dict,
            "label_order": list(label_order),
        }
        atomic_torch_save(self.run_dir / "best_checkpoint.pt", checkpoint)
        prediction = fit.validation_predictions
        prediction_rows = []
        for index, encounter_id in enumerate(prediction.encounter_ids):
            prediction_rows.append(
                {
                    "run_id": self.run_id,
                    "model_id": model_id,
                    "encounter_id": encounter_id,
                    "campaign_id": prediction.campaign_ids[index],
                    "label_scope": "encounter_bag",
                    "label_order": list(label_order),
                    "y_true": prediction.labels[index].astype(int).tolist(),
                    "probability": prediction.probabilities[index].astype(float).tolist(),
                    "logit": prediction.logits[index].astype(float).tolist(),
                }
            )
        atomic_write_jsonl(self.run_dir / "predictions.jsonl", prediction_rows)
        metrics = {
            "best_epoch": fit.best_epoch,
            "best_validation_macro_average_precision": fit.best_validation_macro_ap,
            "epochs_completed": fit.epochs_completed,
            "stopped_early": fit.stopped_early,
            "training_seconds": fit.training_seconds,
            "inference_seconds": prediction.inference_seconds,
            "median_encounter_inference_ms": (
                1000.0 * prediction.inference_seconds / max(len(prediction.encounter_ids), 1)
            ),
            "trainable_parameters": fit.trainable_parameters,
            "positive_class_weights": list(fit.positive_class_weights),
            "determinism": dict(fit.determinism),
        }
        atomic_write_json(self.run_dir / "metrics.json", metrics)
        manifest_path = self.run_dir / "run_manifest.json"
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest.update(
            {
                "status": "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "artifact_sha256": {
                    name: sha256_file(self.run_dir / name)
                    for name in (
                        "configuration.json",
                        "normalizer.json",
                        "history.jsonl",
                        "best_checkpoint.pt",
                        "predictions.jsonl",
                        "metrics.json",
                    )
                },
            }
        )
        atomic_write_json(manifest_path, manifest)

    def _complete_manifest(
        self,
        artifact_names: Sequence[str],
        *,
        outer_test_accessed: bool,
        additional: Mapping[str, Any] | None = None,
    ) -> None:
        manifest_path = self.run_dir / "run_manifest.json"
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest.update(
            {
                "status": "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "outer_test_accessed": bool(outer_test_accessed),
                "artifact_sha256": {
                    name: sha256_file(self.run_dir / name) for name in artifact_names
                },
                **dict(additional or {}),
            }
        )
        atomic_write_json(manifest_path, manifest)

    def save_classical_result(
        self,
        fit: Any,
        *,
        model_id: str,
        outer_test_accessed: bool,
        label_order: Sequence[str] = LABEL_ORDER,
    ) -> None:
        normalizer = (
            dict(fit.tabular_encoder)
            if fit.tabular_encoder is not None
            else {"normalizer": "none", "reason": "training_prevalence_baseline"}
        )
        atomic_write_json(self.run_dir / "normalizer.json", normalizer)
        atomic_write_jsonl(self.run_dir / "history.jsonl", ())
        atomic_torch_save(
            self.run_dir / "best_checkpoint.pt",
            {
                "model_id": str(model_id),
                "run_id": self.run_id,
                "candidate": dict(fit.candidate),
                "model_state": dict(fit.model_state),
                "label_order": list(label_order),
            },
        )
        clipped = np.clip(np.asarray(fit.probabilities), 1.0e-7, 1.0 - 1.0e-7)
        logits = np.log(clipped / (1.0 - clipped))
        prediction_rows = []
        for index, encounter_id in enumerate(fit.encounter_ids):
            prediction_rows.append(
                {
                    "run_id": self.run_id,
                    "model_id": str(model_id),
                    "encounter_id": encounter_id,
                    "campaign_id": fit.campaign_ids[index],
                    "label_scope": "encounter_bag",
                    "label_order": list(label_order),
                    "y_true": fit.labels[index].astype(int).tolist(),
                    "probability": fit.probabilities[index].astype(float).tolist(),
                    "logit": logits[index].astype(float).tolist(),
                }
            )
        atomic_write_jsonl(self.run_dir / "predictions.jsonl", prediction_rows)
        metrics = {
            "candidate": dict(fit.candidate),
            "validation_metrics": evaluation_record(
                fit.labels, fit.probabilities, label_order
            ),
            "best_validation_macro_average_precision": float(
                evaluation_record(fit.labels, fit.probabilities, label_order)[
                    "macro_average_precision"
                ]
            ),
            "best_epoch": None,
            "epochs_completed": 0,
            "training_seconds": None,
            "inference_seconds": None,
            "median_encounter_inference_ms": None,
            "trainable_parameters": _classical_parameter_count(fit.model_state),
        }
        atomic_write_json(self.run_dir / "metrics.json", metrics)
        artifacts = (
            "configuration.json",
            "normalizer.json",
            "history.jsonl",
            "best_checkpoint.pt",
            "predictions.jsonl",
            "metrics.json",
        )
        self._complete_manifest(
            artifacts,
            outer_test_accessed=outer_test_accessed,
        )

    def save_fixed_epoch_result(
        self,
        fit: FixedEpochFitResult,
        prediction: PredictionBundle,
        *,
        model_id: str,
        preprocessing: Mapping[str, Any],
        selection_artifact_sha256: str,
        label_order: Sequence[str] = LABEL_ORDER,
    ) -> None:
        atomic_write_json(self.run_dir / "normalizer.json", dict(preprocessing))
        atomic_write_jsonl(self.run_dir / "history.jsonl", fit.history)
        atomic_torch_save(
            self.run_dir / "best_checkpoint.pt",
            {
                "model_id": str(model_id),
                "run_id": self.run_id,
                "fixed_epochs": fit.epochs_completed,
                "state_dict": fit.state_dict,
                "label_order": list(label_order),
                "selection_artifact_sha256": str(selection_artifact_sha256),
            },
        )
        prediction_rows = []
        for index, encounter_id in enumerate(prediction.encounter_ids):
            prediction_rows.append(
                {
                    "run_id": self.run_id,
                    "model_id": str(model_id),
                    "encounter_id": encounter_id,
                    "campaign_id": prediction.campaign_ids[index],
                    "label_scope": "encounter_bag",
                    "label_order": list(label_order),
                    "y_true": prediction.labels[index].astype(int).tolist(),
                    "probability": prediction.probabilities[index].astype(float).tolist(),
                    "logit": prediction.logits[index].astype(float).tolist(),
                }
            )
        atomic_write_jsonl(self.run_dir / "predictions.jsonl", prediction_rows)
        metrics = {
            "fixed_epochs": fit.epochs_completed,
            "training_seconds": fit.training_seconds,
            "inference_seconds": prediction.inference_seconds,
            "median_encounter_inference_ms": (
                1000.0 * prediction.inference_seconds / max(len(prediction.encounter_ids), 1)
            ),
            "trainable_parameters": fit.trainable_parameters,
            "positive_class_weights": list(fit.positive_class_weights),
            "determinism": dict(fit.determinism),
            "outer_test_metrics": evaluation_record(
                prediction.labels, prediction.probabilities, label_order
            ),
        }
        atomic_write_json(self.run_dir / "metrics.json", metrics)
        artifacts = (
            "configuration.json",
            "normalizer.json",
            "history.jsonl",
            "best_checkpoint.pt",
            "predictions.jsonl",
            "metrics.json",
        )
        self._complete_manifest(
            artifacts,
            outer_test_accessed=True,
            additional={"selection_artifact_sha256": str(selection_artifact_sha256)},
        )


def _classical_parameter_count(model_state: Mapping[str, Any]) -> int:
    count = 0
    for head in model_state.get("heads", ()):
        if head.get("kind") != "logistic_regression":
            continue
        count += int(np.asarray(head["coefficient"]).size)
        count += int(np.asarray(head["intercept"]).size)
    return count


def runtime_environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scikit-learn", "torch", "pandas", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }
