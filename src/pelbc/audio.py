"""Deterministic extraction of the Acoustic LR whistle descriptors."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import fft
from scipy.io import wavfile
from scipy.ndimage import find_objects, gaussian_filter, label, uniform_filter1d
from scipy.signal import windows

from .errors import InputValidationError


FULL_SCALE_INT16 = 32768.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def evenly_spaced_frame_starts(
    sample_count: int,
    frame_length: int,
    hop_length: int,
    max_frames: int,
) -> np.ndarray:
    if min(sample_count, frame_length, hop_length, max_frames) <= 0:
        raise ValueError("sample/frame/hop/max values must all be positive")
    if sample_count < frame_length:
        return np.array([0], dtype=np.int64)
    available = 1 + (sample_count - frame_length) // hop_length
    if available <= max_frames:
        return np.arange(available, dtype=np.int64) * hop_length
    starts = np.linspace(
        0,
        sample_count - frame_length,
        num=max_frames,
        endpoint=True,
        dtype=np.int64,
    )
    return np.unique(starts)


def _safe_ratio_db(numerator: float, denominator: float) -> float | None:
    if numerator <= 0 or denominator <= 0:
        return None
    return float(10.0 * math.log10(numerator / denominator))


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float | None:
    if len(values) == 0:
        return None
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order].astype(np.float64, copy=False)
    total = float(np.sum(sorted_weights))
    if total <= 0:
        return None
    cumulative = np.cumsum(sorted_weights)
    index = int(np.searchsorted(cumulative, quantile * total, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def tonal_salience_spectrogram(
    power: np.ndarray,
    frequencies: np.ndarray,
    audit: Mapping[str, Any],
) -> np.ndarray:
    tiny = np.finfo(np.float64).tiny
    log_power = 10.0 * np.log10(power + tiny)
    smooth = gaussian_filter(
        log_power,
        sigma=(
            float(audit["tonal_smoothing_time_sigma_frames"]),
            float(audit["tonal_smoothing_frequency_sigma_bins"]),
        ),
        mode="nearest",
    )
    bin_hz = float(np.median(np.diff(frequencies)))
    baseline_bins = max(
        3,
        int(round(float(audit["tonal_local_baseline_width_hz"]) / bin_hz)),
    )
    if baseline_bins % 2 == 0:
        baseline_bins += 1
    local_baseline = uniform_filter1d(
        smooth,
        size=baseline_bins,
        axis=1,
        mode="nearest",
    )
    return smooth - local_baseline


def tonal_component_summary(
    salience_spectrogram: np.ndarray,
    frequencies: np.ndarray,
    hop_seconds: float,
    audit: Mapping[str, Any],
    *,
    prefix: str,
    lower_hz: float,
    upper_hz: float,
) -> dict[str, Any]:
    lower = float(lower_hz)
    upper = min(float(upper_hz), float(frequencies[-1]))
    search = (frequencies >= lower) & (frequencies <= upper)
    selected_frequencies = frequencies[search]
    empty = {
        f"{prefix}_component_found": False,
        f"{prefix}_component_candidate_count": 0,
        f"{prefix}_component_frame_fraction": 0.0,
        f"{prefix}_component_duration_seconds": 0.0,
        f"{prefix}_component_pixel_count": 0,
        f"{prefix}_component_score": None,
        f"{prefix}_component_median_salience_db": None,
        f"{prefix}_component_peak_salience_db": None,
        f"{prefix}_component_frequency_q05_hz": None,
        f"{prefix}_component_frequency_q50_hz": None,
        f"{prefix}_component_frequency_q95_hz": None,
        f"{prefix}_component_frequency_q99_hz": None,
        f"{prefix}_component_bandwidth_q90_hz": None,
        f"{prefix}_component_frequency_max_hz": None,
    }
    if not np.any(search):
        return empty

    salience = salience_spectrogram[:, search]
    binary = salience >= float(audit["tonal_salience_threshold_db"])
    components, component_count = label(
        binary,
        structure=np.ones((3, 3), dtype=np.int8),
    )
    empty[f"{prefix}_component_candidate_count"] = int(component_count)
    minimum_frames = max(
        2,
        int(math.ceil(float(audit["tonal_minimum_duration_seconds"]) / hop_seconds)),
    )
    best: dict[str, Any] | None = None
    for component_id, component_slice in enumerate(find_objects(components), start=1):
        if component_slice is None:
            continue
        component_mask = components[component_slice] == component_id
        frame_indices, frequency_indices = np.nonzero(component_mask)
        unique_frames = np.unique(frame_indices)
        if len(unique_frames) < minimum_frames:
            continue
        values = salience[component_slice][component_mask]
        absolute_frequency_indices = frequency_indices + int(component_slice[1].start)
        component_frequencies = selected_frequencies[absolute_frequency_indices]
        pixels_per_frame = len(values) / len(unique_frames)
        median_frequency = float(np.median(component_frequencies))
        high_frequency_penalty = 1.0 + 0.15 * median_frequency / max(upper, 1.0)
        score = (
            float(np.sum(np.maximum(values, 0.0)))
            * math.sqrt(len(unique_frames))
            / math.sqrt(max(pixels_per_frame, 1.0))
            / high_frequency_penalty
        )
        candidate = {
            "score": score,
            "frame_count": int(len(unique_frames)),
            "pixel_count": int(len(values)),
            "frequencies": component_frequencies,
            "salience": np.maximum(values, np.finfo(float).eps),
            "median_salience_db": float(np.median(values)),
            "peak_salience_db": float(np.max(values)),
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    if best is None:
        return empty
    component_frequencies = best.pop("frequencies")
    component_salience = best.pop("salience")
    frequency_quantiles = {
        suffix: _weighted_quantile(component_frequencies, component_salience, quantile)
        for quantile, suffix in (
            (0.05, "q05"),
            (0.50, "q50"),
            (0.95, "q95"),
            (0.99, "q99"),
        )
    }
    return {
        f"{prefix}_component_found": True,
        f"{prefix}_component_candidate_count": int(component_count),
        f"{prefix}_component_frame_fraction": float(
            best["frame_count"] / len(salience_spectrogram)
        ),
        f"{prefix}_component_duration_seconds": float(best["frame_count"] * hop_seconds),
        f"{prefix}_component_pixel_count": int(best["pixel_count"]),
        f"{prefix}_component_score": float(best["score"]),
        f"{prefix}_component_median_salience_db": float(best["median_salience_db"]),
        f"{prefix}_component_peak_salience_db": float(best["peak_salience_db"]),
        f"{prefix}_component_frequency_q05_hz": frequency_quantiles["q05"],
        f"{prefix}_component_frequency_q50_hz": frequency_quantiles["q50"],
        f"{prefix}_component_frequency_q95_hz": frequency_quantiles["q95"],
        f"{prefix}_component_frequency_q99_hz": frequency_quantiles["q99"],
        f"{prefix}_component_bandwidth_q90_hz": float(
            frequency_quantiles["q95"] - frequency_quantiles["q05"]
        ),
        f"{prefix}_component_frequency_max_hz": float(np.max(component_frequencies)),
    }


def extract_waveform_features(
    samples: np.ndarray,
    sample_rate_hz: int,
    audit: Mapping[str, Any],
    flags: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Extract the 14 per-whistle feature families used by Acoustic LR."""

    if samples.ndim != 1:
        raise InputValidationError("audio must be mono")
    if samples.dtype != np.int16:
        raise InputValidationError(
            f"audio must be uncompressed 16-bit PCM; decoded dtype was {samples.dtype}"
        )
    if len(samples) == 0:
        raise InputValidationError("audio contains no samples")
    required_nyquist = float(audit["core_tonal_upper_hz"])
    if sample_rate_hz / 2.0 < required_nyquist:
        raise InputValidationError(
            f"sample rate {sample_rate_hz} Hz is too low; at least "
            f"{int(2 * required_nyquist)} Hz is required"
        )

    sample_count = int(len(samples))
    duration_seconds = sample_count / float(sample_rate_hz)
    integer = np.asarray(samples, dtype=np.int16)
    float_samples = integer.astype(np.float64) / FULL_SCALE_INT16
    absolute_integer = np.abs(integer.astype(np.int32))
    rms_fraction = float(np.sqrt(np.mean(np.square(float_samples), dtype=np.float64)))
    rms_dbfs = float(20.0 * math.log10(max(rms_fraction, np.finfo(float).tiny)))
    dc_fraction = float(np.mean(float_samples, dtype=np.float64))
    exact_clipping_fraction = float(np.mean(absolute_integer >= 32767))
    near_threshold = float(flags["near_clipping_amplitude_fraction"]) * FULL_SCALE_INT16
    near_clipping_fraction = float(np.mean(absolute_integer >= near_threshold))
    zero_fraction = float(np.mean(integer == 0))

    frame_length = max(16, int(round(float(audit["window_seconds"]) * sample_rate_hz)))
    hop_length = max(1, int(round(frame_length * float(audit["hop_fraction"]))))
    starts = evenly_spaced_frame_starts(
        sample_count,
        frame_length,
        hop_length,
        int(audit["max_frames_per_clip"]),
    )
    offsets = np.arange(frame_length, dtype=np.int64)
    if sample_count < frame_length:
        padded = np.zeros(frame_length, dtype=np.float32)
        padded[:sample_count] = integer.astype(np.float32) / FULL_SCALE_INT16
        frames = padded[None, :]
    else:
        frames = integer[starts[:, None] + offsets[None, :]].astype(np.float32)
        frames /= FULL_SCALE_INT16

    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1, dtype=np.float64))
    frame_rms_quantiles = np.quantile(frame_rms, [0.10, 0.50, 0.90])
    temporal_dynamic_db = _safe_ratio_db(
        float(frame_rms_quantiles[2] ** 2),
        float(frame_rms_quantiles[0] ** 2),
    )
    low_activity_threshold = max(float(np.max(frame_rms)) * 0.01, np.finfo(float).tiny)
    low_activity_fraction = float(np.mean(frame_rms < low_activity_threshold))

    frames -= np.mean(frames, axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    window = windows.hann(frame_length, sym=False).astype(np.float32)
    spectrum = fft.rfft(frames * window[None, :], axis=1, workers=1)
    power = np.square(np.abs(spectrum), dtype=np.float64)
    frequencies = fft.rfftfreq(frame_length, d=1.0 / sample_rate_hz)
    salience = tonal_salience_spectrogram(power, frequencies, audit)
    core_tonal = tonal_component_summary(
        salience,
        frequencies,
        hop_length / sample_rate_hz,
        audit,
        prefix="core_tonal",
        lower_hz=float(audit["core_tonal_lower_hz"]),
        upper_hz=float(audit["core_tonal_upper_hz"]),
    )

    quality_flags: list[str] = []
    if exact_clipping_fraction > float(flags["exact_clipping_min_fraction"]):
        quality_flags.append("exact_clipping_detected")
    if near_clipping_fraction >= float(flags["near_clipping_min_fraction"]):
        quality_flags.append("near_clipping_ge_threshold")
    if abs(dc_fraction) >= float(flags["large_dc_offset_fraction_full_scale"]):
        quality_flags.append("large_dc_offset")
    if rms_dbfs <= float(flags["very_low_rms_dbfs"]):
        quality_flags.append("very_low_rms")
    if zero_fraction >= float(flags["large_zero_fraction"]):
        quality_flags.append("large_zero_fraction")
    if core_tonal["core_tonal_component_frame_fraction"] < float(
        flags["low_tonal_component_frame_fraction"]
    ):
        quality_flags.append("low_tonal_component_coverage")
    if duration_seconds > float(flags["nonloop_long_duration_seconds"]):
        quality_flags.append("nonloop_duration_gt_3s")

    features = {
        "duration_seconds": duration_seconds,
        "rms_dbfs": rms_dbfs,
        "temporal_dynamic_db": temporal_dynamic_db,
        "exact_clipping_fraction": exact_clipping_fraction,
        "low_activity_frame_fraction": low_activity_fraction,
        **core_tonal,
    }
    return features, quality_flags


def analyze_wav(
    path: Path,
    audit: Mapping[str, Any],
    flags: Mapping[str, Any],
) -> dict[str, Any]:
    """Read and analyze one mono PCM16 WAV file."""

    path = Path(path)
    try:
        sample_rate_hz, samples = wavfile.read(path, mmap=True)
    except Exception as exc:
        raise InputValidationError(f"cannot decode WAV file {path.name}: {exc}") from exc
    array = np.asarray(samples)
    try:
        features, quality_flags = extract_waveform_features(
            array,
            int(sample_rate_hz),
            audit,
            flags,
        )
    except InputValidationError as exc:
        raise InputValidationError(f"{path.name}: {exc}") from exc
    return {
        "path": path,
        "sha256": sha256_file(path),
        "sample_rate_hz": int(sample_rate_hz),
        "sample_count": int(len(array)),
        "duration_seconds": float(len(array) / int(sample_rate_hz)),
        "features": features,
        "quality_flags": quality_flags,
    }
