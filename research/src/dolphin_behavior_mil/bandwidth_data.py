"""Auditable arbitrary-mel feature registry for bandwidth sensitivity runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .experiment import sha256_file
from .mil_models import MILClassifier, SmallWhistleCNN
from .modeling_data import DatasetRegistry, partition_sha256, read_jsonl


class FlexibleMelWhistleCNN(SmallWhistleCNN):
    """The frozen CNN operations with an explicit non-128 mel input contract."""

    def __init__(self, *, mel_count: int, **kwargs: Any) -> None:
        if int(mel_count) < 1:
            raise ValueError("mel count must be positive")
        self.mel_count = int(mel_count)
        super().__init__(**kwargs)

    def forward_padded(
        self,
        inputs: torch.Tensor,
        time_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if (
            inputs.ndim != 4
            or inputs.shape[1] != 1
            or inputs.shape[2] != self.mel_count
        ):
            raise ValueError(
                f"CNN input must have shape [instance, 1, {self.mel_count}, time]"
            )
        if time_lengths.shape != (len(inputs),):
            raise ValueError("one time length is required per instance")
        if torch.any(time_lengths < 1) or torch.any(time_lengths > inputs.shape[-1]):
            raise ValueError("time lengths must lie within padded input bounds")
        encoded = inputs
        valid_lengths = time_lengths.to(torch.long)
        for block in self.blocks:
            encoded = block(encoded)
            valid_lengths = torch.div(valid_lengths + 1, 2, rounding_mode="floor")
            time_index = torch.arange(encoded.shape[-1], device=encoded.device)[None, :]
            block_mask = time_index < valid_lengths[:, None]
            encoded = encoded * block_mask[:, None, None, :].to(encoded.dtype)
        time_index = torch.arange(encoded.shape[-1], device=encoded.device)[None, :]
        mask = time_index < valid_lengths[:, None]
        weighted = encoded * mask[:, None, None, :].to(encoded.dtype)
        denominator = valid_lengths.to(encoded.dtype) * encoded.shape[2]
        pooled = weighted.sum(dim=(2, 3)) / denominator[:, None].clamp_min(1.0)
        return self.projection(pooled)

    def forward_variable(self, instances: Iterable[torch.Tensor]) -> torch.Tensor:
        values_all = tuple(instances)
        if not values_all:
            raise ValueError("at least one whistle instance is required")
        for value in values_all:
            if (
                value.ndim != 3
                or value.shape[:2] != (1, self.mel_count)
                or value.shape[-1] < 1
            ):
                raise ValueError(
                    f"each instance must have shape [1, {self.mel_count}, time]"
                )
        devices = {str(value.device) for value in values_all}
        if len(devices) != 1:
            raise ValueError("all instances must reside on the same device")
        ordered_indices = sorted(
            range(len(values_all)), key=lambda index: values_all[index].shape[-1]
        )
        output: list[torch.Tensor | None] = [None] * len(values_all)
        maximum = self.encoder_microbatch_max_instances
        for start in range(0, len(ordered_indices), maximum):
            indices = ordered_indices[start : start + maximum]
            values = [values_all[index] for index in indices]
            lengths = torch.tensor(
                [value.shape[-1] for value in values],
                dtype=torch.long,
                device=values[0].device,
            )
            padded = values[0].new_zeros(
                (len(values), 1, self.mel_count, int(lengths.max().item()))
            )
            for row_index, value in enumerate(values):
                padded[row_index, :, :, : value.shape[-1]] = value
            encoded = self.forward_padded(padded, lengths)
            for row_index, original_index in enumerate(indices):
                output[original_index] = encoded[row_index]
        if any(value is None for value in output):
            raise RuntimeError("variable encoder failed to restore instance ordering")
        return torch.stack([value for value in output if value is not None])


def build_bandwidth_hf_lw_gam(
    *,
    mel_count: int,
    base_channels: int,
    embedding_dim: int,
    attention_dim: int,
    dropout: float,
    encoder_microbatch_max_instances: int,
) -> MILClassifier:
    encoder = FlexibleMelWhistleCNN(
        mel_count=mel_count,
        base_channels=base_channels,
        embedding_dim=embedding_dim,
        dropout=dropout,
        encoder_microbatch_max_instances=encoder_microbatch_max_instances,
    )
    return MILClassifier(
        encoder,
        embedding_dim=embedding_dim,
        pooling="label_wise_gated_attention",
        attention_dim=attention_dim,
        dropout=dropout,
    )


@dataclass(frozen=True)
class BandwidthFeatureRegistry:
    base: DatasetRegistry
    manifest_path: Path
    rows_by_whistle: Mapping[str, Mapping[str, Any]]
    mel_count: int
    values_by_whistle: dict[str, np.ndarray] | None = field(
        default=None, repr=False, compare=False
    )

    @classmethod
    def load(
        cls,
        project_root: Path,
        manifest_path: Path,
        *,
        mel_count: int = 192,
        verify_files: bool = True,
        cache_features_in_memory: bool = False,
    ) -> "BandwidthFeatureRegistry":
        root = Path(project_root).resolve()
        base = DatasetRegistry.load(root)
        path = Path(manifest_path).resolve()
        rows = read_jsonl(path)
        by_whistle = {str(row["whistle_id"]): row for row in rows}
        expected = {
            str(row["whistle_id"])
            for values in base.whistles.values()
            for row in values
        }
        if len(rows) != len(by_whistle) or set(by_whistle) != expected:
            raise ValueError("bandwidth manifest must exactly cover 4,126 main instances")
        original = {
            str(row["whistle_id"]): row
            for values in base.whistles.values()
            for row in values
        }
        for whistle_id, row in by_whistle.items():
            shape = tuple(int(value) for value in row["feature_shape"])
            if (
                str(row["encounter_id"]) != str(original[whistle_id]["encounter_id"])
                or str(row["clip_sha256"]) != str(original[whistle_id]["clip_sha256"])
                or shape[0] != int(mel_count)
                or len(shape) != 2
                or str(row["feature_dtype"]) != "float16"
            ):
                raise ValueError(f"invalid bandwidth manifest row: {whistle_id}")
            feature_path = root / str(row["cache_path_relative_to_project"])
            if verify_files:
                if not feature_path.is_file() or sha256_file(feature_path) != str(
                    row["cache_file_sha256"]
                ):
                    raise ValueError(f"bandwidth cache hash mismatch: {whistle_id}")
                value = np.load(feature_path, allow_pickle=False)
                if (
                    value.shape != shape
                    or value.dtype != np.float16
                    or not np.all(np.isfinite(value))
                ):
                    raise ValueError(f"invalid bandwidth cache content: {whistle_id}")
        return cls(
            base=base,
            manifest_path=path,
            rows_by_whistle=by_whistle,
            mel_count=int(mel_count),
            values_by_whistle={} if cache_features_in_memory else None,
        )

    @property
    def project_root(self) -> Path:
        return self.base.project_root

    @property
    def version_dir(self) -> Path:
        return self.base.version_dir

    @property
    def encounters(self) -> Mapping[str, Mapping[str, Any]]:
        return self.base.encounters

    @property
    def labels(self) -> Mapping[str, np.ndarray]:
        return self.base.labels

    @property
    def whistles(self) -> Mapping[str, tuple[Mapping[str, Any], ...]]:
        return self.base.whistles

    @property
    def campaign_by_encounter(self) -> Mapping[str, str]:
        return self.base.campaign_by_encounter

    @property
    def outer_fold_by_encounter(self) -> Mapping[str, str]:
        return self.base.outer_fold_by_encounter

    @property
    def encounter_ids(self) -> tuple[str, ...]:
        return self.base.encounter_ids

    def validate_encounter_ids(self, values: Iterable[str]) -> tuple[str, ...]:
        return self.base.validate_encounter_ids(values)

    def outer_split(self, outer_fold_id: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return self.base.outer_split(outer_fold_id)

    def feature_path(self, whistle: Mapping[str, Any]) -> Path:
        row = self.rows_by_whistle[str(whistle["whistle_id"])]
        return self.project_root / str(row["cache_path_relative_to_project"])

    def load_feature(self, whistle: Mapping[str, Any]) -> np.ndarray:
        whistle_id = str(whistle["whistle_id"])
        if self.values_by_whistle is not None and whistle_id in self.values_by_whistle:
            return self.values_by_whistle[whistle_id]
        row = self.rows_by_whistle[whistle_id]
        value = np.load(self.feature_path(whistle), allow_pickle=False)
        expected_shape = tuple(int(number) for number in row["feature_shape"])
        if (
            value.shape != expected_shape
            or value.ndim != 2
            or value.shape[0] != self.mel_count
            or value.dtype != np.float16
            or not np.all(np.isfinite(value))
        ):
            raise ValueError(f"invalid bandwidth feature: {whistle_id}")
        result = np.asarray(value, dtype=np.float32)
        if self.values_by_whistle is not None:
            self.values_by_whistle[whistle_id] = result
        return result


@dataclass(frozen=True)
class MelFeatureNormalizer:
    mean: np.ndarray
    standard_deviation: np.ndarray
    pixel_count_per_mel: int
    training_partition_sha256: str
    training_encounter_count: int
    training_instance_count: int
    minimum_standard_deviation: float = 1.0e-6

    @classmethod
    def fit(
        cls,
        registry: BandwidthFeatureRegistry,
        encounter_ids: Iterable[str],
        *,
        minimum_standard_deviation: float = 1.0e-6,
    ) -> "MelFeatureNormalizer":
        ids = registry.validate_encounter_ids(encounter_ids)
        if minimum_standard_deviation <= 0:
            raise ValueError("minimum standard deviation must be positive")
        total = np.zeros(registry.mel_count, dtype=np.float64)
        squares = np.zeros(registry.mel_count, dtype=np.float64)
        pixel_count = 0
        instance_count = 0
        for encounter_id in ids:
            for whistle in registry.whistles[encounter_id]:
                feature = registry.load_feature(whistle).astype(np.float64, copy=False)
                total += np.sum(feature, axis=1, dtype=np.float64)
                squares += np.sum(np.square(feature), axis=1, dtype=np.float64)
                pixel_count += feature.shape[1]
                instance_count += 1
        if pixel_count < 2:
            raise ValueError("normalizer requires at least two time-frequency pixels")
        mean = total / pixel_count
        variance = np.maximum(squares / pixel_count - np.square(mean), 0.0)
        standard_deviation = np.maximum(
            np.sqrt(variance), float(minimum_standard_deviation)
        )
        return cls(
            mean=mean.astype(np.float32),
            standard_deviation=standard_deviation.astype(np.float32),
            pixel_count_per_mel=int(pixel_count),
            training_partition_sha256=partition_sha256(ids),
            training_encounter_count=len(ids),
            training_instance_count=instance_count,
            minimum_standard_deviation=float(minimum_standard_deviation),
        )

    def transform(self, feature: np.ndarray) -> np.ndarray:
        value = np.asarray(feature, dtype=np.float32)
        if value.ndim != 2 or value.shape[0] != len(self.mean):
            raise ValueError("feature mel dimension differs from normalizer")
        return np.asarray(
            (value - self.mean[:, None]) / self.standard_deviation[:, None],
            dtype=np.float32,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalizer": "per_mel_zscore_bandwidth_sensitivity_v1",
            "mel_count": len(self.mean),
            "mean": self.mean.astype(float).tolist(),
            "standard_deviation": self.standard_deviation.astype(float).tolist(),
            "pixel_count_per_mel": self.pixel_count_per_mel,
            "training_partition_sha256": self.training_partition_sha256,
            "training_encounter_count": self.training_encounter_count,
            "training_instance_count": self.training_instance_count,
            "minimum_standard_deviation": self.minimum_standard_deviation,
        }
