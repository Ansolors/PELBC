"""Leakage-safe encounter bags and fold-local feature preparation for modeling."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover - training is an optional dependency
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


LABEL_ORDER = ("Feeding", "Travelling", "Milling", "Socializing")
PRIMARY_ENCOUNTER_COUNT = 234
MAIN_INSTANCE_COUNT = 4126


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def partition_sha256(encounter_ids: Iterable[str]) -> str:
    canonical = "\n".join(sorted(str(value) for value in encounter_ids)) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_unique(rows: Sequence[Mapping[str, Any]], key: str, table: str) -> None:
    values = [str(row[key]) for row in rows]
    if len(values) != len(set(values)):
        raise ValueError(f"{table} contains duplicate {key} values")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetRegistry:
    project_root: Path
    version_dir: Path
    encounters: Mapping[str, Mapping[str, Any]]
    labels: Mapping[str, np.ndarray]
    whistles: Mapping[str, tuple[Mapping[str, Any], ...]]
    environments: Mapping[str, Mapping[str, Any]]
    campaign_by_encounter: Mapping[str, str]
    outer_fold_by_encounter: Mapping[str, str]
    inner_rows: tuple[Mapping[str, Any], ...]
    label_order: tuple[str, ...] = LABEL_ORDER
    feature_values_by_whistle: dict[str, np.ndarray] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @classmethod
    def load(
        cls,
        project_root: Path,
        *,
        dataset_version: str = "0.4.0",
        validate_cache_files: bool = True,
        cache_features_in_memory: bool = False,
    ) -> "DatasetRegistry":
        root = Path(project_root).resolve()
        version_dir = root / "data" / "processed" / f"v{dataset_version}"
        if not version_dir.is_dir():
            raise FileNotFoundError(version_dir)

        encounter_rows = read_jsonl(version_dir / "encounters.jsonl")
        label_rows = read_jsonl(version_dir / "labels.jsonl")
        whistle_rows = read_jsonl(version_dir / "whistles.jsonl")
        environment_rows = read_jsonl(version_dir / "environment_flat.jsonl")
        outer_rows = read_jsonl(version_dir / "outer_fold_assignments.jsonl")
        inner_rows = read_jsonl(version_dir / "inner_fold_assignments.jsonl")

        _require_unique(encounter_rows, "encounter_id", "encounters")
        _require_unique(label_rows, "encounter_id", "labels")
        _require_unique(whistle_rows, "whistle_id", "whistles")
        _require_unique(environment_rows, "source_csv_row_number", "environment")

        labels: dict[str, np.ndarray] = {}
        for row in label_rows:
            if tuple(row["primary_label_order"]) != LABEL_ORDER:
                raise ValueError("primary label order differs from the frozen contract")
            vector = np.asarray(row["primary_label_vector"], dtype=np.float32)
            if vector.shape != (len(LABEL_ORDER),) or not np.all(
                (vector == 0) | (vector == 1)
            ):
                raise ValueError(f"invalid label vector for {row['encounter_id']}")
            labels[str(row["encounter_id"])] = vector

        if len(labels) != PRIMARY_ENCOUNTER_COUNT:
            raise ValueError(
                f"expected {PRIMARY_ENCOUNTER_COUNT} primary encounters, got {len(labels)}"
            )

        encounters_all = {str(row["encounter_id"]): row for row in encounter_rows}
        encounters = {key: encounters_all[key] for key in labels}
        environment_by_row = {
            int(row["source_csv_row_number"]): row for row in environment_rows
        }
        environments: dict[str, Mapping[str, Any]] = {}
        for encounter_id, encounter in encounters.items():
            source_row = int(encounter["selected_environment_row"])
            if source_row not in environment_by_row:
                raise ValueError(f"missing selected environment row for {encounter_id}")
            environment = environment_by_row[source_row]
            if environment["record_id"] != encounter["selected_environment_record_id"]:
                raise ValueError(f"selected environment identity mismatch for {encounter_id}")
            environments[encounter_id] = environment

        whistles_mutable: dict[str, list[Mapping[str, Any]]] = {
            encounter_id: [] for encounter_id in labels
        }
        for row in whistle_rows:
            encounter_id = str(row["encounter_id"])
            if encounter_id not in labels:
                continue
            if not bool(row.get("primary_analysis_cohort")):
                raise ValueError(f"primary whistle flag missing for {row['whistle_id']}")
            if not bool(row.get("main_discrete_whistle_instance")):
                continue
            feature = row.get("feature_cache")
            if not isinstance(feature, Mapping):
                raise ValueError(f"missing feature cache record for {row['whistle_id']}")
            cache_path = root / str(feature["cache_path_relative_to_project"])
            if validate_cache_files and not cache_path.is_file():
                raise FileNotFoundError(cache_path)
            whistles_mutable[encounter_id].append(row)

        whistles: dict[str, tuple[Mapping[str, Any], ...]] = {}
        for encounter_id, values in whistles_mutable.items():
            values.sort(key=lambda row: (int(row["sequence_index"]), str(row["whistle_id"])))
            indices = [int(row["sequence_index"]) for row in values]
            if not values or len(indices) != len(set(indices)):
                raise ValueError(f"invalid main whistle sequence for {encounter_id}")
            expected = int(encounters[encounter_id]["main_discrete_whistle_count"])
            if len(values) != expected:
                raise ValueError(
                    f"main whistle count mismatch for {encounter_id}: {len(values)} != {expected}"
                )
            whistles[encounter_id] = tuple(values)

        total_instances = sum(len(values) for values in whistles.values())
        if total_instances != MAIN_INSTANCE_COUNT:
            raise ValueError(
                f"expected {MAIN_INSTANCE_COUNT} main instances, got {total_instances}"
            )

        primary_outer = [row for row in outer_rows if bool(row.get("primary_234"))]
        _require_unique(primary_outer, "encounter_id", "outer assignments")
        outer_ids = {str(row["encounter_id"]) for row in primary_outer}
        if outer_ids != set(labels):
            raise ValueError("outer assignments do not exactly cover the primary cohort")
        campaign_by_encounter = {
            str(row["encounter_id"]): str(row["campaign_id"]) for row in primary_outer
        }
        outer_fold_by_encounter = {
            str(row["encounter_id"]): str(row["outer_test_fold_id"])
            for row in primary_outer
        }

        return cls(
            project_root=root,
            version_dir=version_dir,
            encounters=encounters,
            labels=labels,
            whistles=whistles,
            environments=environments,
            campaign_by_encounter=campaign_by_encounter,
            outer_fold_by_encounter=outer_fold_by_encounter,
            inner_rows=tuple(inner_rows),
            feature_values_by_whistle={} if cache_features_in_memory else None,
        )

    @property
    def encounter_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.labels))

    def validate_encounter_ids(self, encounter_ids: Iterable[str]) -> tuple[str, ...]:
        values = tuple(str(value) for value in encounter_ids)
        if not values:
            raise ValueError("an encounter partition cannot be empty")
        if len(values) != len(set(values)):
            raise ValueError("an encounter partition contains duplicates")
        unknown = sorted(set(values) - set(self.labels))
        if unknown:
            raise KeyError(f"unknown primary encounter IDs: {unknown[:3]}")
        return tuple(sorted(values))

    def outer_split(self, outer_fold_id: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        fold = str(outer_fold_id)
        known = sorted(set(self.outer_fold_by_encounter.values()))
        if fold not in known:
            raise KeyError(f"unknown outer fold {fold}; expected one of {known}")
        test_ids = tuple(
            sorted(
                encounter_id
                for encounter_id, assigned in self.outer_fold_by_encounter.items()
                if assigned == fold
            )
        )
        train_ids = tuple(sorted(set(self.labels) - set(test_ids)))
        self.assert_campaign_disjoint(train_ids, test_ids)
        return train_ids, test_ids

    def inner_split(
        self,
        outer_fold_id: str,
        inner_validation_fold_id: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        outer_train, _ = self.outer_split(outer_fold_id)
        rows = [
            row
            for row in self.inner_rows
            if str(row["outer_fold_id"]) == str(outer_fold_id)
        ]
        assigned = {str(row["encounter_id"]): str(row["inner_validation_fold_id"]) for row in rows}
        if set(assigned) != set(outer_train):
            raise ValueError(f"inner registry does not exactly cover {outer_fold_id} training data")
        known = sorted(set(assigned.values()))
        requested = str(inner_validation_fold_id)
        if requested not in known:
            raise KeyError(f"unknown inner fold {requested}; expected one of {known}")
        validation_ids = tuple(sorted(key for key, value in assigned.items() if value == requested))
        train_ids = tuple(sorted(set(outer_train) - set(validation_ids)))
        self.assert_campaign_disjoint(train_ids, validation_ids)
        return train_ids, validation_ids

    def assert_campaign_disjoint(
        self,
        first_ids: Iterable[str],
        second_ids: Iterable[str],
    ) -> None:
        first = self.validate_encounter_ids(first_ids)
        second = self.validate_encounter_ids(second_ids)
        overlap_ids = set(first) & set(second)
        if overlap_ids:
            raise ValueError(f"encounter leakage across partitions: {sorted(overlap_ids)[:3]}")
        first_campaigns = {self.campaign_by_encounter[value] for value in first}
        second_campaigns = {self.campaign_by_encounter[value] for value in second}
        overlap = sorted(first_campaigns & second_campaigns)
        if overlap:
            raise ValueError(f"campaign leakage across partitions: {overlap[:3]}")

    def feature_path(self, whistle: Mapping[str, Any]) -> Path:
        return self.project_root / str(whistle["feature_cache"]["cache_path_relative_to_project"])

    def load_feature(self, whistle: Mapping[str, Any]) -> np.ndarray:
        whistle_id = str(whistle["whistle_id"])
        if (
            self.feature_values_by_whistle is not None
            and whistle_id in self.feature_values_by_whistle
        ):
            return np.asarray(self.feature_values_by_whistle[whistle_id], dtype=np.float32)
        path = self.feature_path(whistle)
        feature = np.load(path, allow_pickle=False)
        expected_shape = tuple(int(value) for value in whistle["feature_cache"]["feature_shape"])
        if feature.shape != expected_shape or feature.ndim != 2 or feature.shape[0] != 128:
            raise ValueError(f"feature shape mismatch for {whistle['whistle_id']}")
        if feature.dtype != np.float16:
            raise ValueError(f"feature dtype mismatch for {whistle['whistle_id']}")
        if not np.all(np.isfinite(feature)):
            raise ValueError(f"nonfinite feature for {whistle['whistle_id']}")
        if self.feature_values_by_whistle is not None:
            self.feature_values_by_whistle[whistle_id] = feature
        return np.asarray(feature, dtype=np.float32)

    def preload_feature_values(self) -> dict[str, int]:
        if self.feature_values_by_whistle is None:
            raise RuntimeError("DatasetRegistry was not configured for in-memory features")
        for encounter_id in self.encounter_ids:
            for whistle in self.whistles[encounter_id]:
                self.load_feature(whistle)
        return {
            "instance_count": len(self.feature_values_by_whistle),
            "bytes": int(
                sum(value.nbytes for value in self.feature_values_by_whistle.values())
            ),
        }


@dataclass(frozen=True)
class FeatureNormalizer:
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
        registry: DatasetRegistry,
        encounter_ids: Iterable[str],
        *,
        minimum_standard_deviation: float = 1.0e-6,
    ) -> "FeatureNormalizer":
        ids = registry.validate_encounter_ids(encounter_ids)
        if minimum_standard_deviation <= 0:
            raise ValueError("minimum_standard_deviation must be positive")
        total = np.zeros(128, dtype=np.float64)
        total_squares = np.zeros(128, dtype=np.float64)
        count = 0
        instance_count = 0
        for encounter_id in ids:
            for whistle in registry.whistles[encounter_id]:
                feature = registry.load_feature(whistle).astype(np.float64, copy=False)
                total += np.sum(feature, axis=1, dtype=np.float64)
                total_squares += np.sum(np.square(feature), axis=1, dtype=np.float64)
                count += feature.shape[1]
                instance_count += 1
        if count < 2:
            raise ValueError("normalizer requires at least two time-frequency pixels per mel")
        mean = total / count
        variance = np.maximum(total_squares / count - np.square(mean), 0.0)
        standard_deviation = np.sqrt(variance)
        standard_deviation = np.maximum(standard_deviation, minimum_standard_deviation)
        return cls(
            mean=mean.astype(np.float32),
            standard_deviation=standard_deviation.astype(np.float32),
            pixel_count_per_mel=int(count),
            training_partition_sha256=partition_sha256(ids),
            training_encounter_count=len(ids),
            training_instance_count=instance_count,
            minimum_standard_deviation=float(minimum_standard_deviation),
        )

    def transform(self, feature: np.ndarray) -> np.ndarray:
        value = np.asarray(feature, dtype=np.float32)
        if value.ndim != 2 or value.shape[0] != len(self.mean):
            raise ValueError("feature must have shape [mel, time] matching the normalizer")
        return np.asarray(
            (value - self.mean[:, None]) / self.standard_deviation[:, None],
            dtype=np.float32,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalizer": "per_mel_zscore_v1",
            "mean": self.mean.astype(float).tolist(),
            "standard_deviation": self.standard_deviation.astype(float).tolist(),
            "pixel_count_per_mel": self.pixel_count_per_mel,
            "training_partition_sha256": self.training_partition_sha256,
            "training_encounter_count": self.training_encounter_count,
            "training_instance_count": self.training_instance_count,
            "minimum_standard_deviation": self.minimum_standard_deviation,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "FeatureNormalizer":
        mean = np.asarray(record["mean"], dtype=np.float32)
        standard_deviation = np.asarray(record["standard_deviation"], dtype=np.float32)
        if mean.shape != (128,) or standard_deviation.shape != (128,):
            raise ValueError("serialized normalizer must contain 128 mel statistics")
        if not np.all(np.isfinite(mean)) or np.any(standard_deviation <= 0):
            raise ValueError("serialized normalizer contains invalid statistics")
        return cls(
            mean=mean,
            standard_deviation=standard_deviation,
            pixel_count_per_mel=int(record["pixel_count_per_mel"]),
            training_partition_sha256=str(record["training_partition_sha256"]),
            training_encounter_count=int(record["training_encounter_count"]),
            training_instance_count=int(record["training_instance_count"]),
            minimum_standard_deviation=float(record["minimum_standard_deviation"]),
        )


@dataclass(frozen=True)
class NormalizedFeatureCache:
    training_partition_sha256: str
    values_by_whistle: Mapping[str, np.ndarray]
    encounter_ids: tuple[str, ...]

    @classmethod
    def build(
        cls,
        registry: DatasetRegistry,
        normalizer: FeatureNormalizer,
        encounter_ids: Iterable[str],
    ) -> "NormalizedFeatureCache":
        ids = registry.validate_encounter_ids(encounter_ids)
        values: dict[str, np.ndarray] = {}
        for encounter_id in ids:
            for whistle in registry.whistles[encounter_id]:
                whistle_id = str(whistle["whistle_id"])
                values[whistle_id] = normalizer.transform(registry.load_feature(whistle))
        return cls(
            training_partition_sha256=normalizer.training_partition_sha256,
            values_by_whistle=values,
            encounter_ids=ids,
        )

    @property
    def bytes(self) -> int:
        return int(sum(value.nbytes for value in self.values_by_whistle.values()))


def _stable_rng(*parts: object) -> np.random.Generator:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)
    return np.random.default_rng(seed)


class MILBagDataset(Dataset):
    """Load complete encounter bags; optional training subsampling never creates labels."""

    def __init__(
        self,
        registry: DatasetRegistry,
        encounter_ids: Iterable[str],
        normalizer: FeatureNormalizer,
        *,
        training: bool,
        maximum_instances_per_bag: int | None = None,
        seed: int = 0,
        augmentation: Mapping[str, Any] | None = None,
        normalized_feature_cache: NormalizedFeatureCache | None = None,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for MILBagDataset")
        self.registry = registry
        self.encounter_ids = registry.validate_encounter_ids(encounter_ids)
        self.normalizer = normalizer
        self.training = bool(training)
        if maximum_instances_per_bag is not None and maximum_instances_per_bag < 1:
            raise ValueError("maximum_instances_per_bag must be positive")
        if not self.training and maximum_instances_per_bag is not None:
            raise ValueError("evaluation bags must not be randomly capped")
        self.maximum_instances_per_bag = maximum_instances_per_bag
        self.seed = int(seed)
        self.augmentation = dict(augmentation or {})
        self.normalized_feature_cache = normalized_feature_cache
        if (
            normalized_feature_cache is not None
            and normalized_feature_cache.training_partition_sha256
            != normalizer.training_partition_sha256
        ):
            raise ValueError("normalized feature cache was built with a different normalizer")
        if normalized_feature_cache is not None:
            required_whistles = {
                str(row["whistle_id"])
                for encounter_id in self.encounter_ids
                for row in registry.whistles[encounter_id]
            }
            if not required_whistles <= set(normalized_feature_cache.values_by_whistle):
                raise ValueError("normalized feature cache does not cover this dataset")
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.encounter_ids)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _selected_whistles(self, encounter_id: str) -> tuple[Mapping[str, Any], ...]:
        values = self.registry.whistles[encounter_id]
        maximum = self.maximum_instances_per_bag
        if not self.training or maximum is None or len(values) <= maximum:
            return values
        rng = _stable_rng("bag_subset", self.seed, self.epoch, encounter_id)
        indices = np.sort(rng.choice(len(values), size=maximum, replace=False))
        return tuple(values[int(index)] for index in indices)

    def _augment(self, feature: np.ndarray, whistle_id: str) -> np.ndarray:
        if not self.training:
            return feature
        frequency_count = int(self.augmentation.get("frequency_mask_count", 0))
        frequency_width = int(self.augmentation.get("frequency_mask_max_bins", 0))
        time_count = int(self.augmentation.get("time_mask_count", 0))
        time_fraction = float(self.augmentation.get("time_mask_max_fraction", 0.0))
        fill = float(self.augmentation.get("mask_fill_after_normalization", 0.0))
        if frequency_count <= 0 and time_count <= 0:
            return feature
        result = feature.copy()
        rng = _stable_rng("augmentation", self.seed, self.epoch, whistle_id)
        for _ in range(frequency_count):
            width = int(rng.integers(0, min(frequency_width, result.shape[0]) + 1))
            if width:
                start = int(rng.integers(0, result.shape[0] - width + 1))
                result[start : start + width, :] = fill
        maximum_time_width = min(
            result.shape[1], int(np.floor(result.shape[1] * max(time_fraction, 0.0)))
        )
        for _ in range(time_count):
            width = int(rng.integers(0, maximum_time_width + 1))
            if width:
                start = int(rng.integers(0, result.shape[1] - width + 1))
                result[:, start : start + width] = fill
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        encounter_id = self.encounter_ids[index]
        selected = self._selected_whistles(encounter_id)
        features = []
        whistle_ids = []
        sequence_indices = []
        for whistle in selected:
            whistle_id = str(whistle["whistle_id"])
            if self.normalized_feature_cache is None:
                feature = self.registry.load_feature(whistle)
                feature = self.normalizer.transform(feature)
            else:
                feature = self.normalized_feature_cache.values_by_whistle[whistle_id]
            feature = self._augment(feature, str(whistle["whistle_id"]))
            features.append(torch.from_numpy(feature[None, :, :]))
            whistle_ids.append(whistle_id)
            sequence_indices.append(int(whistle["sequence_index"]))
        return {
            "encounter_id": encounter_id,
            "campaign_id": self.registry.campaign_by_encounter[encounter_id],
            "features": tuple(features),
            "whistle_ids": tuple(whistle_ids),
            "sequence_indices": tuple(sequence_indices),
            "label": torch.from_numpy(self.registry.labels[encounter_id].copy()),
            "available_instance_count": len(self.registry.whistles[encounter_id]),
        }


@dataclass(frozen=True)
class MILBatch:
    instances: tuple[Any, ...]
    bag_index: Any
    labels: Any
    encounter_ids: tuple[str, ...]
    campaign_ids: tuple[str, ...]
    whistle_ids: tuple[str, ...]
    sequence_indices: tuple[int, ...]
    available_instance_counts: tuple[int, ...]

    @property
    def bag_count(self) -> int:
        return len(self.encounter_ids)

    def to(self, device: Any) -> "MILBatch":
        return MILBatch(
            instances=tuple(value.to(device) for value in self.instances),
            bag_index=self.bag_index.to(device),
            labels=self.labels.to(device),
            encounter_ids=self.encounter_ids,
            campaign_ids=self.campaign_ids,
            whistle_ids=self.whistle_ids,
            sequence_indices=self.sequence_indices,
            available_instance_counts=self.available_instance_counts,
        )


def collate_mil_bags(samples: Sequence[Mapping[str, Any]]) -> MILBatch:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for MIL collation")
    if not samples:
        raise ValueError("cannot collate an empty batch")
    instances = []
    bag_indices = []
    whistle_ids = []
    sequence_indices = []
    for bag_index, sample in enumerate(samples):
        values = tuple(sample["features"])
        if not values:
            raise ValueError("every encounter bag must contain at least one instance")
        instances.extend(values)
        bag_indices.extend([bag_index] * len(values))
        whistle_ids.extend(str(value) for value in sample["whistle_ids"])
        sequence_indices.extend(int(value) for value in sample["sequence_indices"])
    return MILBatch(
        instances=tuple(instances),
        bag_index=torch.tensor(bag_indices, dtype=torch.long),
        labels=torch.stack([sample["label"] for sample in samples]).to(torch.float32),
        encounter_ids=tuple(str(sample["encounter_id"]) for sample in samples),
        campaign_ids=tuple(str(sample["campaign_id"]) for sample in samples),
        whistle_ids=tuple(whistle_ids),
        sequence_indices=tuple(sequence_indices),
        available_instance_counts=tuple(
            int(sample["available_instance_count"]) for sample in samples
        ),
    )


@dataclass(frozen=True)
class FrozenEmbeddingRegistry:
    encoder_id: str
    embedding_dimension: int
    rows_by_whistle: Mapping[str, Mapping[str, Any]]
    manifest_path: Path
    project_root: Path
    values_by_whistle: Mapping[str, np.ndarray] | None = None

    @classmethod
    def load(
        cls,
        project_root: Path,
        manifest_path: Path,
        dataset_registry: DatasetRegistry,
        *,
        verify_files: bool = True,
        preload_values: bool = False,
    ) -> "FrozenEmbeddingRegistry":
        root = Path(project_root).resolve()
        repository_root = (
            root.parent
            if root.name == "experiments" and (root.parent / "data").is_dir()
            else root
        )
        path = Path(manifest_path).resolve()
        rows = read_jsonl(path)
        required = {
            "whistle_id",
            "encounter_id",
            "encoder_id",
            "encoder_revision",
            "embedding_path_relative_to_project",
            "embedding_shape",
            "source_clip_sha256",
            "embedding_sha256",
            "input_sample_rate_hz",
            "retained_nyquist_hz",
            "license",
        }
        if not rows:
            raise ValueError("embedding manifest is empty")
        _require_unique(rows, "whistle_id", "embedding manifest")
        encoder_ids = {str(row.get("encoder_id")) for row in rows}
        if len(encoder_ids) != 1:
            raise ValueError("embedding manifest must contain exactly one encoder_id")
        encoder_revisions = {str(row.get("encoder_revision")) for row in rows}
        if len(encoder_revisions) != 1:
            raise ValueError("embedding manifest must contain exactly one encoder revision")
        licenses = {str(row.get("license")) for row in rows}
        if len(licenses) != 1 or not next(iter(licenses)):
            raise ValueError("embedding manifest must contain one nonempty license")
        expected_whistles = {
            str(row["whistle_id"])
            for values in dataset_registry.whistles.values()
            for row in values
        }
        row_ids = {str(row["whistle_id"]) for row in rows}
        if row_ids != expected_whistles:
            raise ValueError("embedding manifest must exactly cover all 4,126 main instances")
        dimension: int | None = None
        by_whistle: dict[str, Mapping[str, Any]] = {}
        preloaded: dict[str, np.ndarray] = {}
        for row in rows:
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(f"embedding row is missing fields: {missing}")
            whistle_id = str(row["whistle_id"])
            encounter_id = str(row["encounter_id"])
            if encounter_id not in dataset_registry.whistles:
                raise ValueError(f"unknown embedding encounter for {whistle_id}")
            whistle_lookup = {
                str(value["whistle_id"]): value
                for value in dataset_registry.whistles[encounter_id]
            }
            if whistle_id not in whistle_lookup:
                raise ValueError(f"embedding encounter mismatch for {whistle_id}")
            source_sha = str(row["source_clip_sha256"])
            if source_sha != str(whistle_lookup[whistle_id]["clip_sha256"]):
                raise ValueError(f"embedding source hash mismatch for {whistle_id}")
            shape = tuple(int(value) for value in row["embedding_shape"])
            if len(shape) != 1 or shape[0] < 1:
                raise ValueError(f"embedding must be a vector for {whistle_id}")
            if dimension is None:
                dimension = shape[0]
            elif shape[0] != dimension:
                raise ValueError("embedding dimensions are inconsistent")
            relative_embedding_path = Path(str(row["embedding_path_relative_to_project"]))
            if relative_embedding_path.is_absolute():
                raise ValueError(f"embedding path must be project-relative for {whistle_id}")
            embedding_path = (root / relative_embedding_path).resolve()
            if (
                embedding_path != repository_root
                and repository_root not in embedding_path.parents
            ):
                raise ValueError(
                    f"embedding path escapes repository root for {whistle_id}"
                )
            input_rate = int(row["input_sample_rate_hz"])
            nyquist = int(row["retained_nyquist_hz"])
            if input_rate < 1 or nyquist < 1 or nyquist > input_rate / 2:
                raise ValueError(f"invalid embedding bandwidth fields for {whistle_id}")
            embedding_sha = str(row["embedding_sha256"])
            if len(embedding_sha) != 64 or any(
                character not in "0123456789abcdef" for character in embedding_sha
            ):
                raise ValueError(f"invalid embedding SHA-256 for {whistle_id}")
            if verify_files:
                if not embedding_path.is_file():
                    raise FileNotFoundError(embedding_path)
                if _sha256_file(embedding_path) != embedding_sha:
                    raise ValueError(f"embedding file hash mismatch for {whistle_id}")
                value = np.load(embedding_path, allow_pickle=False)
                if (
                    value.shape != shape
                    or value.dtype != np.float32
                    or not np.all(np.isfinite(value))
                ):
                    raise ValueError(f"invalid embedding file content for {whistle_id}")
                if preload_values:
                    preloaded[whistle_id] = np.asarray(value, dtype=np.float32)
            elif preload_values:
                if not embedding_path.is_file():
                    raise FileNotFoundError(embedding_path)
                value = np.load(embedding_path, allow_pickle=False)
                if (
                    value.shape != shape
                    or value.dtype != np.float32
                    or not np.all(np.isfinite(value))
                ):
                    raise ValueError(f"invalid embedding file content for {whistle_id}")
                preloaded[whistle_id] = np.asarray(value, dtype=np.float32)
            by_whistle[whistle_id] = row
        assert dimension is not None
        return cls(
            encoder_id=next(iter(encoder_ids)),
            embedding_dimension=dimension,
            rows_by_whistle=by_whistle,
            manifest_path=path,
            project_root=root,
            values_by_whistle=preloaded if preload_values else None,
        )

    def load_embedding(self, whistle_id: str) -> np.ndarray:
        if whistle_id not in self.rows_by_whistle:
            raise KeyError(f"missing frozen embedding for {whistle_id}")
        if self.values_by_whistle is not None:
            return self.values_by_whistle[whistle_id]
        row = self.rows_by_whistle[whistle_id]
        path = self.project_root / str(row["embedding_path_relative_to_project"])
        value = np.load(path, allow_pickle=False)
        expected = tuple(int(number) for number in row["embedding_shape"])
        if value.shape != expected or value.shape != (self.embedding_dimension,):
            raise ValueError(f"frozen embedding shape mismatch for {whistle_id}")
        if not np.issubdtype(value.dtype, np.floating) or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid frozen embedding values for {whistle_id}")
        return np.asarray(value, dtype=np.float32)


class FrozenEmbeddingBagDataset(Dataset):
    """Encounter bags backed by a complete, auditable frozen-embedding manifest."""

    def __init__(
        self,
        registry: DatasetRegistry,
        embedding_registry: FrozenEmbeddingRegistry,
        encounter_ids: Iterable[str],
        *,
        training: bool,
        maximum_instances_per_bag: int | None = None,
        seed: int = 0,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for frozen embedding bags")
        self.registry = registry
        self.embedding_registry = embedding_registry
        self.encounter_ids = registry.validate_encounter_ids(encounter_ids)
        self.training = bool(training)
        if maximum_instances_per_bag is not None and maximum_instances_per_bag < 1:
            raise ValueError("maximum_instances_per_bag must be positive")
        if not training and maximum_instances_per_bag is not None:
            raise ValueError("evaluation embedding bags must be complete")
        self.maximum_instances_per_bag = maximum_instances_per_bag
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.encounter_ids)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _selected_whistles(self, encounter_id: str) -> tuple[Mapping[str, Any], ...]:
        values = self.registry.whistles[encounter_id]
        maximum = self.maximum_instances_per_bag
        if not self.training or maximum is None or len(values) <= maximum:
            return values
        rng = _stable_rng("frozen_bag_subset", self.seed, self.epoch, encounter_id)
        indices = np.sort(rng.choice(len(values), size=maximum, replace=False))
        return tuple(values[int(index)] for index in indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        encounter_id = self.encounter_ids[index]
        selected = self._selected_whistles(encounter_id)
        whistle_ids = tuple(str(row["whistle_id"]) for row in selected)
        return {
            "encounter_id": encounter_id,
            "campaign_id": self.registry.campaign_by_encounter[encounter_id],
            "features": tuple(
                torch.from_numpy(self.embedding_registry.load_embedding(whistle_id))
                for whistle_id in whistle_ids
            ),
            "whistle_ids": whistle_ids,
            "sequence_indices": tuple(int(row["sequence_index"]) for row in selected),
            "label": torch.from_numpy(self.registry.labels[encounter_id].copy()),
            "available_instance_count": len(self.registry.whistles[encounter_id]),
        }
