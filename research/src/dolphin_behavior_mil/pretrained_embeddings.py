"""Auditable, frozen whistle embeddings from PANNs and AVES-bio."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import types
from typing import Any, Callable, Mapping, Sequence
import wave

import numpy as np
from scipy.signal import resample_poly

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None


EXPECTED_MAIN_INSTANCE_COUNT = 4126
ProgressCallback = Callable[[int, int, Mapping[str, Any]], None]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_pretrained_config(
    project_root: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    path = Path(config_path or root / "configs/pretrained_encoders_v1.toml").resolve()
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    if config["protocol"]["protocol_id"] != "pretrained_encoders_v1":
        raise ValueError("unexpected pretrained encoder protocol")
    if int(config["protocol"]["expected_instance_count"]) != EXPECTED_MAIN_INSTANCE_COUNT:
        raise ValueError("pretrained protocol does not target all 4,126 main instances")
    return config


def _validated_project_path(project_root: Path, relative_path: str) -> Path:
    root = Path(project_root).resolve()
    repository_root = (
        root.parent
        if root.name == "experiments" and (root.parent / "data").is_dir()
        else root
    )
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"project asset path must be relative: {relative}")
    resolved = (root / relative).resolve()
    if resolved != repository_root and repository_root not in resolved.parents:
        raise ValueError(f"project asset escapes repository root: {relative}")
    return resolved


def validate_file_asset(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int | None = None,
) -> dict[str, Any]:
    value = Path(path)
    if not value.is_file():
        raise FileNotFoundError(value)
    size = value.stat().st_size
    if expected_bytes is not None and size != int(expected_bytes):
        raise ValueError(f"asset byte count mismatch for {value}: {size} != {expected_bytes}")
    actual_sha256 = sha256_file(value)
    if actual_sha256 != str(expected_sha256):
        raise ValueError(f"asset SHA-256 mismatch for {value}")
    return {"path": str(value), "bytes": size, "sha256": actual_sha256}


def validate_encoder_assets(
    project_root: Path,
    encoder_config: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    records: dict[str, Any] = {}
    weight_path = _validated_project_path(root, str(encoder_config["weight_path"]))
    records["weight"] = validate_file_asset(
        weight_path,
        expected_sha256=str(encoder_config["weight_sha256"]),
        expected_bytes=int(encoder_config["weight_bytes"]),
    )
    license_path = _validated_project_path(root, str(encoder_config["source_license_path"]))
    records["license"] = validate_file_asset(
        license_path,
        expected_sha256=str(encoder_config["source_license_sha256"]),
    )
    source_path = _validated_project_path(root, str(encoder_config["source_path"]))
    if not source_path.is_dir():
        raise FileNotFoundError(source_path)
    records["source"] = {
        "path": str(source_path),
        "revision": str(encoder_config["source_revision"]),
    }
    if "model_config_path" in encoder_config:
        model_config_path = _validated_project_path(
            root, str(encoder_config["model_config_path"])
        )
        records["model_config"] = validate_file_asset(
            model_config_path,
            expected_sha256=str(encoder_config["model_config_sha256"]),
            expected_bytes=int(encoder_config["model_config_bytes"]),
        )
    return records


@dataclass(frozen=True)
class DecodedWaveform:
    samples: np.ndarray
    sample_rate_hz: int
    source_sha256: str
    source_frame_count: int
    dc_offset_before_correction: float


def decode_pcm16_mono(path: Path) -> DecodedWaveform:
    """Decode a RIFF/WAVE mono PCM16 file and apply frozen per-clip DC correction."""

    value = Path(path)
    blob = value.read_bytes()
    source_sha256 = sha256_bytes(blob)
    if blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise ValueError(f"not a little-endian RIFF/WAVE file: {value}")
    with wave.open(io.BytesIO(blob), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        compression = handle.getcomptype()
        raw = handle.readframes(frame_count)
    if channels != 1:
        raise ValueError(f"pretrained encoder input must be mono: {value}")
    if sample_width != 2 or compression != "NONE":
        raise ValueError(f"pretrained encoder input must be uncompressed PCM16: {value}")
    pcm = np.frombuffer(raw, dtype="<i2")
    if pcm.size != frame_count:
        raise ValueError(f"decoded frame count mismatch: {value}")
    samples = pcm.astype(np.float32) / np.float32(32768.0)
    dc_offset = float(np.mean(samples, dtype=np.float64))
    samples = np.asarray(samples - np.float32(dc_offset), dtype=np.float32)
    if not np.all(np.isfinite(samples)):
        raise ValueError(f"nonfinite samples after decoding: {value}")
    return DecodedWaveform(
        samples=samples,
        sample_rate_hz=int(sample_rate),
        source_sha256=source_sha256,
        source_frame_count=int(frame_count),
        dc_offset_before_correction=dc_offset,
    )


def resample_float32(
    samples: np.ndarray,
    source_sample_rate_hz: int,
    target_sample_rate_hz: int,
) -> np.ndarray:
    value = np.asarray(samples, dtype=np.float32)
    if value.ndim != 1 or value.size < 1:
        raise ValueError("waveform must be a nonempty vector")
    source_rate = int(source_sample_rate_hz)
    target_rate = int(target_sample_rate_hz)
    if source_rate < 1 or target_rate < 1:
        raise ValueError("sample rates must be positive")
    if source_rate == target_rate:
        return value.copy()
    divisor = math.gcd(source_rate, target_rate)
    output = resample_poly(
        value,
        up=target_rate // divisor,
        down=source_rate // divisor,
    )
    output = np.asarray(output, dtype=np.float32)
    if output.ndim != 1 or output.size < 1 or not np.all(np.isfinite(output)):
        raise ValueError("resampling produced an invalid waveform")
    return output


def prepare_encoder_waveform(
    decoded: DecodedWaveform,
    *,
    target_sample_rate_hz: int,
    minimum_input_samples: int,
    allow_right_padding: bool,
) -> tuple[np.ndarray, int]:
    samples = resample_float32(
        decoded.samples,
        decoded.sample_rate_hz,
        int(target_sample_rate_hz),
    )
    minimum = int(minimum_input_samples)
    if samples.size < minimum:
        if not allow_right_padding:
            raise ValueError(
                f"waveform has {samples.size} samples, below encoder minimum {minimum}"
            )
        padding = minimum - samples.size
        samples = np.pad(samples, (0, padding), mode="constant")
    else:
        padding = 0
    return np.asarray(samples, dtype=np.float32), int(padding)


class FrozenWhistleExtractor:
    def __init__(
        self,
        project_root: Path,
        encoder_config: Mapping[str, Any],
        *,
        device: str,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for pretrained embeddings")
        self.project_root = Path(project_root).resolve()
        self.config = dict(encoder_config)
        self.device = torch.device(device)
        self.embedding_dimension = int(self.config["embedding_dimension"])
        self.input_sample_rate_hz = int(self.config["input_sample_rate_hz"])
        self.retained_nyquist_hz = int(self.config["retained_nyquist_hz"])
        self.effective_feature_max_hz = int(self.config["effective_feature_max_hz"])

    def prepare(self, decoded: DecodedWaveform) -> tuple[np.ndarray, int]:
        return prepare_encoder_waveform(
            decoded,
            target_sample_rate_hz=self.input_sample_rate_hz,
            minimum_input_samples=int(self.config["minimum_input_samples"]),
            allow_right_padding=str(self.config["encoder_key"]) == "panns",
        )

    def extract_prepared(self, waveform: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def extract(self, decoded: DecodedWaveform) -> tuple[np.ndarray, dict[str, int]]:
        waveform, padding = self.prepare(decoded)
        embedding = np.asarray(self.extract_prepared(waveform), dtype=np.float32)
        if embedding.shape != (self.embedding_dimension,):
            raise ValueError(
                f"{self.config['encoder_id']} returned {embedding.shape}, expected "
                f"({self.embedding_dimension},)"
            )
        if not np.all(np.isfinite(embedding)):
            raise ValueError(f"nonfinite embedding from {self.config['encoder_id']}")
        return embedding, {
            "resampled_sample_count_before_padding": int(waveform.size - padding),
            "right_padding_samples": padding,
            "encoder_input_sample_count": int(waveform.size),
        }


def _load_locked_panns_models(source_root: Path) -> Any:
    """Import PANNs models without executing its home-directory downloading __init__."""

    package_name = "_dolphin_mil_locked_panns"
    model_name = f"{package_name}.models"
    expected_package_path = Path(source_root) / "panns_inference"
    existing = sys.modules.get(package_name)
    if existing is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(expected_package_path)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    elif list(getattr(existing, "__path__", [])) != [str(expected_package_path)]:
        raise RuntimeError("a different locked PANNs source is already imported")
    return importlib.import_module(model_name)


class PannsCnn14Extractor(FrozenWhistleExtractor):
    def __init__(
        self,
        project_root: Path,
        encoder_config: Mapping[str, Any],
        *,
        device: str,
    ) -> None:
        super().__init__(project_root, encoder_config, device=device)
        source_root = _validated_project_path(
            self.project_root, str(self.config["source_path"])
        )
        models = _load_locked_panns_models(source_root)
        self.model = models.Cnn14(
            sample_rate=self.input_sample_rate_hz,
            window_size=int(self.config["window_size_samples"]),
            hop_size=int(self.config["hop_size_samples"]),
            mel_bins=int(self.config["mel_bins"]),
            fmin=int(self.config["mel_min_hz"]),
            fmax=int(self.config["mel_max_hz"]),
            classes_num=int(self.config["audioset_class_count"]),
        )
        weight_path = _validated_project_path(
            self.project_root, str(self.config["weight_path"])
        )
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
        if set(checkpoint) < {"model"}:
            raise ValueError("PANNs checkpoint has no model state")
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.to(self.device).eval().requires_grad_(False)

    def extract_prepared(self, waveform: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).unsqueeze(0)
        with torch.inference_mode():
            output = self.model(tensor.to(self.device))
        embedding = output["embedding"]
        if tuple(embedding.shape) != (1, self.embedding_dimension):
            raise ValueError(f"unexpected PANNs embedding shape: {tuple(embedding.shape)}")
        return embedding[0].detach().to("cpu", torch.float32).numpy()


class AvesBioExtractor(FrozenWhistleExtractor):
    def __init__(
        self,
        project_root: Path,
        encoder_config: Mapping[str, Any],
        *,
        device: str,
    ) -> None:
        super().__init__(project_root, encoder_config, device=device)
        try:
            from torchaudio.models import wav2vec2_model
        except ModuleNotFoundError as error:  # pragma: no cover
            raise RuntimeError("TorchAudio is required for AVES embeddings") from error
        config_path = _validated_project_path(
            self.project_root, str(self.config["model_config_path"])
        )
        with config_path.open(encoding="utf-8") as handle:
            model_config = json.load(handle)
        self.model = wav2vec2_model(**model_config, aux_num_out=None)
        weight_path = _validated_project_path(
            self.project_root, str(self.config["weight_path"])
        )
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval().requires_grad_(False)

    def extract_prepared(self, waveform: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).unsqueeze(0)
        with torch.inference_mode():
            layers = self.model.extract_features(tensor.to(self.device))[0]
        selected = layers[int(self.config["layer"])]
        if selected.ndim != 3 or selected.shape[0] != 1 or selected.shape[2] != self.embedding_dimension:
            raise ValueError(f"unexpected AVES frame shape: {tuple(selected.shape)}")
        embedding = torch.mean(selected, dim=1)[0]
        return embedding.detach().to("cpu", torch.float32).numpy()


def build_extractor(
    project_root: Path,
    encoder_config: Mapping[str, Any],
    *,
    device: str,
) -> FrozenWhistleExtractor:
    key = str(encoder_config["encoder_key"])
    if key == "panns":
        return PannsCnn14Extractor(project_root, encoder_config, device=device)
    if key == "aves":
        return AvesBioExtractor(project_root, encoder_config, device=device)
    raise KeyError(f"unsupported pretrained encoder: {key}")


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, np.ascontiguousarray(value), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(destination)


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)


def main_whistle_rows(dataset_registry: Any) -> list[Mapping[str, Any]]:
    rows = [
        row
        for encounter_id in dataset_registry.encounter_ids
        for row in dataset_registry.whistles[encounter_id]
    ]
    rows.sort(key=lambda row: (str(row["encounter_id"]), int(row["sequence_index"])))
    if len(rows) != EXPECTED_MAIN_INSTANCE_COUNT:
        raise ValueError(f"expected 4,126 main whistles, got {len(rows)}")
    return rows


def embedding_output_path(
    project_root: Path,
    cache_root_relative: str,
    encoder_key: str,
    whistle_row: Mapping[str, Any],
) -> Path:
    cache_root = _validated_project_path(project_root, cache_root_relative)
    return (
        cache_root
        / encoder_key
        / str(whistle_row["encounter_id"])
        / f"{whistle_row['whistle_id']}.npy"
    )


def _load_valid_existing_embedding(path: Path, expected_dimension: int) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.float32 or value.shape != (expected_dimension,):
        raise ValueError(f"invalid existing embedding shape/dtype: {path}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"nonfinite existing embedding: {path}")
    return np.asarray(value, dtype=np.float32)


def cache_encoder_embeddings(
    *,
    project_root: Path,
    protocol_config: Mapping[str, Any],
    encoder_config: Mapping[str, Any],
    whistle_rows: Sequence[Mapping[str, Any]],
    extractor: FrozenWhistleExtractor,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(project_root).resolve()
    if len(whistle_rows) != int(protocol_config["protocol"]["expected_instance_count"]):
        raise ValueError("embedding cache requires the complete main whistle cohort")
    start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    padding_count = 0
    resumed_count = 0
    computed_count = 0
    norms: list[float] = []
    for index, whistle in enumerate(whistle_rows, start=1):
        source_path = root.parent.parent / str(
            whistle["clip_path_relative_to_workspace"]
        )
        output_path = embedding_output_path(
            root,
            str(protocol_config["storage"]["cache_root"]),
            str(encoder_config["encoder_key"]),
            whistle,
        )
        decoded = decode_pcm16_mono(source_path)
        if decoded.source_sha256 != str(whistle["clip_sha256"]):
            raise ValueError(f"source clip SHA-256 mismatch for {whistle['whistle_id']}")
        prepared, padding = extractor.prepare(decoded)
        if output_path.is_file() and not force:
            embedding = _load_valid_existing_embedding(
                output_path, int(encoder_config["embedding_dimension"])
            )
            resumed_count += 1
        else:
            embedding = extractor.extract_prepared(prepared)
            embedding = np.asarray(embedding, dtype=np.float32)
            if embedding.shape != (int(encoder_config["embedding_dimension"]),):
                raise ValueError(f"invalid embedding shape for {whistle['whistle_id']}")
            if not np.all(np.isfinite(embedding)):
                raise ValueError(f"nonfinite embedding for {whistle['whistle_id']}")
            atomic_save_npy(output_path, embedding)
            computed_count += 1
        if padding:
            padding_count += 1
        relative_output = output_path.relative_to(root)
        embedding_hash = sha256_file(output_path)
        norms.append(float(np.linalg.norm(embedding.astype(np.float64))))
        row = {
            "whistle_id": str(whistle["whistle_id"]),
            "encounter_id": str(whistle["encounter_id"]),
            "sequence_index": int(whistle["sequence_index"]),
            "encoder_id": str(encoder_config["encoder_id"]),
            "encoder_revision": str(encoder_config["encoder_revision"]),
            "embedding_path_relative_to_project": str(relative_output),
            "embedding_shape": [int(encoder_config["embedding_dimension"])],
            "embedding_dtype": "float32",
            "source_clip_sha256": decoded.source_sha256,
            "embedding_sha256": embedding_hash,
            "input_sample_rate_hz": int(encoder_config["input_sample_rate_hz"]),
            "retained_nyquist_hz": int(encoder_config["retained_nyquist_hz"]),
            "effective_feature_max_hz": int(encoder_config["effective_feature_max_hz"]),
            "license": str(encoder_config["license"]),
            "source_sample_rate_hz": decoded.sample_rate_hz,
            "source_frame_count": decoded.source_frame_count,
            "source_duration_seconds": decoded.source_frame_count / decoded.sample_rate_hz,
            "dc_offset_before_correction": decoded.dc_offset_before_correction,
            "resampled_sample_count_before_padding": int(prepared.size - padding),
            "right_padding_samples": int(padding),
            "encoder_input_sample_count": int(prepared.size),
            "inference_batch_size": 1,
            "time_pooling": str(encoder_config["pooling"]),
        }
        rows.append(row)
        if progress is not None:
            progress(index, len(whistle_rows), row)

    elapsed = time.perf_counter() - start
    summary = {
        "protocol_id": str(protocol_config["protocol"]["protocol_id"]),
        "source_dataset_version": str(protocol_config["protocol"]["source_dataset_version"]),
        "encoder_key": str(encoder_config["encoder_key"]),
        "encoder_id": str(encoder_config["encoder_id"]),
        "encoder_revision": str(encoder_config["encoder_revision"]),
        "embedding_count": len(rows),
        "embedding_dimension": int(encoder_config["embedding_dimension"]),
        "embedding_dtype": "float32",
        "computed_count_this_run": computed_count,
        "resumed_count_this_run": resumed_count,
        "right_padded_clip_count": padding_count,
        "elapsed_seconds": elapsed,
        "mean_seconds_per_clip_including_io_and_hashing": elapsed / len(rows),
        "embedding_l2_norm": {
            "minimum": min(norms),
            "median": float(np.median(norms)),
            "maximum": max(norms),
        },
        "input_sample_rate_hz": int(encoder_config["input_sample_rate_hz"]),
        "retained_nyquist_hz": int(encoder_config["retained_nyquist_hz"]),
        "effective_feature_max_hz": int(encoder_config["effective_feature_max_hz"]),
        "canonical_device": str(extractor.device),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return rows, summary
