"""Deterministic encounter-level training loop for variable-length MIL bags."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .evaluation import evaluation_record, macro_average_precision
from .mil_models import MILClassifier, trainable_parameter_count
from .modeling_data import MILBagDataset, MILBatch, collate_mil_bags, partition_sha256


def seed_everything(seed: int, *, deterministic_algorithms: bool) -> dict[str, Any]:
    value = int(seed)
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.use_deterministic_algorithms(bool(deterministic_algorithms), warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(deterministic_algorithms)
    return {
        "seed": value,
        "deterministic_algorithms_requested": bool(deterministic_algorithms),
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }


def positive_class_weights(labels: np.ndarray) -> np.ndarray:
    truth = np.asarray(labels, dtype=np.float64)
    if truth.ndim != 2 or not len(truth) or not np.all((truth == 0) | (truth == 1)):
        raise ValueError("labels must be a nonempty binary matrix")
    positive = np.sum(truth, axis=0)
    negative = len(truth) - positive
    if np.any(positive == 0) or np.any(negative == 0):
        raise ValueError("every training label needs both positive and negative encounters")
    return np.asarray(negative / positive, dtype=np.float32)


@dataclass(frozen=True)
class NeuralTrainingConfig:
    learning_rate: float
    weight_decay: float
    encounter_batch_size: int
    maximum_epochs: int
    early_stopping_patience: int
    early_stopping_minimum_delta: float
    gradient_clip_norm: float
    seed: int
    deterministic_algorithms: bool = True
    data_loader_workers: int = 0

    def validate(self) -> None:
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer rates are invalid")
        if min(
            self.encounter_batch_size,
            self.maximum_epochs,
            self.early_stopping_patience,
        ) < 1:
            raise ValueError("batch, epoch and patience values must be positive")
        if self.early_stopping_minimum_delta < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("early stopping delta and gradient clip are invalid")
        if self.data_loader_workers < 0:
            raise ValueError("data_loader_workers cannot be negative")


@dataclass(frozen=True)
class PredictionBundle:
    encounter_ids: tuple[str, ...]
    campaign_ids: tuple[str, ...]
    labels: np.ndarray
    probabilities: np.ndarray
    logits: np.ndarray
    inference_seconds: float


@dataclass(frozen=True)
class FitResult:
    best_epoch: int
    best_validation_macro_ap: float
    epochs_completed: int
    stopped_early: bool
    training_seconds: float
    trainable_parameters: int
    history: tuple[Mapping[str, Any], ...]
    determinism: Mapping[str, Any]
    positive_class_weights: tuple[float, ...]
    validation_predictions: PredictionBundle
    best_state_dict: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class FixedEpochFitResult:
    epochs_completed: int
    training_seconds: float
    trainable_parameters: int
    history: tuple[Mapping[str, Any], ...]
    determinism: Mapping[str, Any]
    positive_class_weights: tuple[float, ...]
    state_dict: Mapping[str, torch.Tensor]


def dataset_preprocessing_record(
    train_dataset: Any,
    validation_dataset: Any | None = None,
) -> dict[str, Any]:
    """Validate fold-local preprocessing or a shared frozen embedding registry."""

    training_partition = partition_sha256(train_dataset.encounter_ids)
    normalizer = getattr(train_dataset, "normalizer", None)
    embedding_registry = getattr(train_dataset, "embedding_registry", None)
    if normalizer is not None:
        if normalizer.training_partition_sha256 != training_partition:
            raise ValueError("training normalizer partition identity mismatch")
        if validation_dataset is not None:
            validation_normalizer = getattr(validation_dataset, "normalizer", None)
            if (
                validation_normalizer is None
                or validation_normalizer.training_partition_sha256 != training_partition
            ):
                raise ValueError(
                    "training and validation must use the same train-fitted normalizer"
                )
        return normalizer.to_dict()
    if embedding_registry is not None:
        if validation_dataset is not None:
            validation_registry = getattr(validation_dataset, "embedding_registry", None)
            if (
                validation_registry is None
                or validation_registry.encoder_id != embedding_registry.encoder_id
                or validation_registry.embedding_dimension
                != embedding_registry.embedding_dimension
                or validation_registry.manifest_path.resolve()
                != embedding_registry.manifest_path.resolve()
            ):
                raise ValueError("training and validation use different frozen embeddings")
        return {
            "normalizer": "none_for_frozen_embeddings",
            "encoder_id": embedding_registry.encoder_id,
            "embedding_dimension": embedding_registry.embedding_dimension,
            "embedding_manifest": str(embedding_registry.manifest_path),
            "training_partition_sha256": training_partition,
        }
    raise TypeError("unrecognized MIL dataset preprocessing contract")


def _loader(
    dataset: MILBagDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        collate_fn=collate_mil_bags,
        generator=generator,
        persistent_workers=bool(workers),
    )


def predict_mil(
    model: MILClassifier,
    dataset: MILBagDataset,
    *,
    batch_size: int,
    device: torch.device,
    data_loader_workers: int = 0,
) -> PredictionBundle:
    loader = _loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        workers=data_loader_workers,
    )
    model.eval()
    encounter_ids: list[str] = []
    campaign_ids: list[str] = []
    labels = []
    logits = []
    start = time.perf_counter()
    with torch.no_grad():
        for raw_batch in loader:
            batch: MILBatch = raw_batch.to(device)
            output = model(batch.instances, batch.bag_index, batch.bag_count)
            encounter_ids.extend(batch.encounter_ids)
            campaign_ids.extend(batch.campaign_ids)
            labels.append(batch.labels.detach().cpu().numpy())
            logits.append(output.logits.detach().cpu().numpy())
    elapsed = time.perf_counter() - start
    label_matrix = np.concatenate(labels, axis=0).astype(np.int8)
    logit_matrix = np.concatenate(logits, axis=0).astype(np.float64)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logit_matrix, -50.0, 50.0)))
    return PredictionBundle(
        encounter_ids=tuple(encounter_ids),
        campaign_ids=tuple(campaign_ids),
        labels=label_matrix,
        probabilities=probability,
        logits=logit_matrix,
        inference_seconds=float(elapsed),
    )


def train_one_epoch(
    model: MILClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    device: torch.device,
    gradient_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    encounter_count = 0
    for raw_batch in loader:
        batch: MILBatch = raw_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.instances, batch.bag_index, batch.bag_count)
        loss = criterion(output.logits, batch.labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("training loss became nonfinite")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(gradient_clip_norm))
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * batch.bag_count
        encounter_count += batch.bag_count
    if not encounter_count:
        raise RuntimeError("training loader yielded no encounters")
    return total_loss / encounter_count


def fit_neural_model(
    model: MILClassifier,
    train_dataset: MILBagDataset,
    validation_dataset: MILBagDataset,
    config: NeuralTrainingConfig,
    *,
    device: str | torch.device = "cpu",
) -> FitResult:
    config.validate()
    if not train_dataset.training or validation_dataset.training:
        raise ValueError("fit requires a training dataset and an uncapped evaluation dataset")
    train_dataset.registry.assert_campaign_disjoint(
        train_dataset.encounter_ids, validation_dataset.encounter_ids
    )
    dataset_preprocessing_record(train_dataset, validation_dataset)

    determinism = seed_everything(
        config.seed, deterministic_algorithms=config.deterministic_algorithms
    )
    resolved_device = torch.device(device)
    model.to(resolved_device)
    training_labels = np.stack(
        [train_dataset.registry.labels[value] for value in train_dataset.encounter_ids]
    )
    weights_array = positive_class_weights(training_labels)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(weights_array, dtype=torch.float32, device=resolved_device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loader = _loader(
        train_dataset,
        batch_size=config.encounter_batch_size,
        shuffle=True,
        seed=config.seed,
        workers=config.data_loader_workers,
    )

    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    start = time.perf_counter()
    for epoch in range(1, config.maximum_epochs + 1):
        train_dataset.set_epoch(epoch)
        epoch_start = time.perf_counter()
        training_loss = train_one_epoch(
            model,
            loader,
            optimizer,
            criterion,
            device=resolved_device,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        prediction = predict_mil(
            model,
            validation_dataset,
            batch_size=config.encounter_batch_size,
            device=resolved_device,
            data_loader_workers=config.data_loader_workers,
        )
        score = macro_average_precision(prediction.labels, prediction.probabilities)
        improved = score > best_score + config.early_stopping_minimum_delta
        if improved:
            best_score = float(score)
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(training_loss),
                "validation_macro_average_precision": float(score),
                "validation_metrics": evaluation_record(
                    prediction.labels,
                    prediction.probabilities,
                    train_dataset.registry.label_order,
                ),
                "epoch_seconds": float(time.perf_counter() - epoch_start),
                "improved": bool(improved),
            }
        )
        if epochs_without_improvement >= config.early_stopping_patience:
            break
    elapsed = time.perf_counter() - start
    if best_state is None or best_epoch < 1:
        raise RuntimeError("training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    final_prediction = predict_mil(
        model,
        validation_dataset,
        batch_size=config.encounter_batch_size,
        device=resolved_device,
        data_loader_workers=config.data_loader_workers,
    )
    return FitResult(
        best_epoch=best_epoch,
        best_validation_macro_ap=float(best_score),
        epochs_completed=len(history),
        stopped_early=len(history) < config.maximum_epochs,
        training_seconds=float(elapsed),
        trainable_parameters=trainable_parameter_count(model),
        history=tuple(history),
        determinism=determinism,
        positive_class_weights=tuple(float(value) for value in weights_array),
        validation_predictions=final_prediction,
        best_state_dict=best_state,
    )


def fit_neural_fixed_epochs(
    model: MILClassifier,
    train_dataset: Any,
    config: NeuralTrainingConfig,
    *,
    epochs: int,
    device: str | torch.device = "cpu",
) -> FixedEpochFitResult:
    """Fit on a complete outer-training pool for a preselected epoch count."""

    config.validate()
    if not train_dataset.training:
        raise ValueError("fixed-epoch fit requires a training dataset")
    if int(epochs) < 1:
        raise ValueError("fixed epoch count must be positive")
    dataset_preprocessing_record(train_dataset)
    determinism = seed_everything(
        config.seed, deterministic_algorithms=config.deterministic_algorithms
    )
    resolved_device = torch.device(device)
    model.to(resolved_device)
    training_labels = np.stack(
        [train_dataset.registry.labels[value] for value in train_dataset.encounter_ids]
    )
    weights_array = positive_class_weights(training_labels)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(weights_array, dtype=torch.float32, device=resolved_device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loader = _loader(
        train_dataset,
        batch_size=config.encounter_batch_size,
        shuffle=True,
        seed=config.seed,
        workers=config.data_loader_workers,
    )
    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        train_dataset.set_epoch(epoch)
        epoch_start = time.perf_counter()
        training_loss = train_one_epoch(
            model,
            loader,
            optimizer,
            criterion,
            device=resolved_device,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(training_loss),
                "epoch_seconds": float(time.perf_counter() - epoch_start),
            }
        )
    elapsed = time.perf_counter() - start
    state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    return FixedEpochFitResult(
        epochs_completed=int(epochs),
        training_seconds=float(elapsed),
        trainable_parameters=trainable_parameter_count(model),
        history=tuple(history),
        determinism=determinism,
        positive_class_weights=tuple(float(value) for value in weights_array),
        state_dict=state,
    )


def training_config_dict(config: NeuralTrainingConfig) -> dict[str, Any]:
    return asdict(config)
