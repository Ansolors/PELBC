"""Deterministic waveform-quality and native-band spectral audit utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import fft
from scipy.io import wavfile
from scipy.ndimage import find_objects, gaussian_filter, label, uniform_filter1d
from scipy.signal import windows


FULL_SCALE_INT16 = 32768.0


def evenly_spaced_frame_starts(
    sample_count: int,
    frame_length: int,
    hop_length: int,
    max_frames: int,
) -> np.ndarray:
    """Return deterministic frame starts spanning the complete clip."""

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


def _frequency_quantile(
    frequencies: np.ndarray,
    energy: np.ndarray,
    quantile: float,
) -> float | None:
    total = float(np.sum(energy, dtype=np.float64))
    if not math.isfinite(total) or total <= 0:
        return None
    cumulative = np.cumsum(energy, dtype=np.float64)
    index = int(np.searchsorted(cumulative, quantile * total, side="left"))
    index = min(index, len(frequencies) - 1)
    return float(frequencies[index])


def _safe_ratio_db(numerator: float, denominator: float) -> float | None:
    if numerator <= 0 or denominator <= 0:
        return None
    return float(10.0 * math.log10(numerator / denominator))


def _normalized_entropy(energy: np.ndarray) -> float | None:
    total = float(np.sum(energy, dtype=np.float64))
    if total <= 0:
        return None
    probabilities = energy.astype(np.float64, copy=False) / total
    probabilities = probabilities[probabilities > 0]
    if len(probabilities) <= 1:
        return 0.0
    return float(-np.sum(probabilities * np.log(probabilities)) / math.log(len(energy)))


def _retention(
    frequencies: np.ndarray,
    energy: np.ndarray,
    cutoff_hz: float,
) -> float | None:
    total = float(np.sum(energy, dtype=np.float64))
    if total <= 0:
        return None
    retained = float(np.sum(energy[frequencies <= cutoff_hz], dtype=np.float64))
    return retained / total


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
    """Return local narrow-band salience without artificial analysis-band edges."""

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
    """Find the strongest continuous narrow-band component in a whitened spectrum."""

    lower = float(lower_hz)
    upper = min(float(upper_hz), float(frequencies[-1]))
    search = (frequencies >= lower) & (frequencies <= upper)
    selected_frequencies = frequencies[search]
    if not np.any(search):
        return {
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
    salience = salience_spectrogram[:, search]
    binary = salience >= float(audit["tonal_salience_threshold_db"])
    components, component_count = label(
        binary,
        structure=np.ones((3, 3), dtype=np.int8),
    )
    minimum_frames = max(
        2,
        int(math.ceil(float(audit["tonal_minimum_duration_seconds"]) / hop_seconds)),
    )
    best: dict[str, Any] | None = None
    for component_id, component_slice in enumerate(
        find_objects(components),
        start=1,
    ):
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

    empty = {
        f"{prefix}_component_found": False,
        f"{prefix}_component_candidate_count": int(component_count),
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
    if best is None:
        return empty
    component_frequencies = best.pop("frequencies")
    component_salience = best.pop("salience")
    frequency_quantiles = {
        suffix: _weighted_quantile(component_frequencies, component_salience, q)
        for q, suffix in ((0.05, "q05"), (0.50, "q50"), (0.95, "q95"), (0.99, "q99"))
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


def _finite_float(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def audit_waveform(
    samples: np.ndarray,
    sample_rate_hz: int,
    audit: Mapping[str, Any],
    flags: Mapping[str, Any],
    *,
    loop_type: int | None,
    duration_seconds: float,
) -> dict[str, Any]:
    """Audit one mono PCM16 waveform without assigning biological labels."""

    if samples.ndim != 1:
        raise ValueError("audio must be mono")
    if samples.dtype != np.int16:
        raise ValueError(f"expected PCM16 int16, got {samples.dtype}")
    if len(samples) == 0:
        raise ValueError("audio contains no samples")

    sample_count = int(len(samples))
    integer = np.asarray(samples, dtype=np.int16)
    float_samples = integer.astype(np.float64) / FULL_SCALE_INT16
    absolute_integer = np.abs(integer.astype(np.int32))
    peak_fraction = float(np.max(absolute_integer) / FULL_SCALE_INT16)
    rms_fraction = float(np.sqrt(np.mean(np.square(float_samples), dtype=np.float64)))
    rms_dbfs = float(20.0 * math.log10(max(rms_fraction, np.finfo(float).tiny)))
    dc_fraction = float(np.mean(float_samples, dtype=np.float64))
    exact_clipping_fraction = float(np.mean(absolute_integer >= 32767))
    near_threshold = float(flags["near_clipping_amplitude_fraction"]) * FULL_SCALE_INT16
    near_clipping_fraction = float(np.mean(absolute_integer >= near_threshold))
    zero_fraction = float(np.mean(integer == 0))

    cap = int(audit["amplitude_percentile_sample_cap"])
    if sample_count <= cap:
        amplitude_sample = float_samples
    else:
        indices = np.linspace(0, sample_count - 1, cap, dtype=np.int64)
        amplitude_sample = float_samples[indices]
    absolute_sample = np.abs(amplitude_sample)
    amplitude_quantiles = np.quantile(
        absolute_sample,
        [0.50, 0.90, 0.95, 0.99],
    )

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
    baseline = np.quantile(
        power,
        float(audit["spectral_baseline_quantile"]),
        axis=0,
    )
    raw_energy = np.mean(power, axis=0, dtype=np.float64)
    excess_frames = np.maximum(power - baseline[None, :], 0.0)
    excess_energy = np.mean(excess_frames, axis=0, dtype=np.float64)

    lower_hz = float(audit["analysis_lower_hz"])
    analysis_mask = frequencies >= lower_hz
    analysis_frequencies = frequencies[analysis_mask]
    raw_analysis = raw_energy[analysis_mask]
    excess_analysis = excess_energy[analysis_mask]
    baseline_analysis = baseline[analysis_mask]

    spectral: dict[str, Any] = {
        "audit_frame_length_samples": frame_length,
        "audit_frame_count": int(len(frames)),
        "audit_frequency_resolution_hz": float(sample_rate_hz / frame_length),
        "raw_spectral_entropy": _normalized_entropy(raw_analysis),
        "excess_spectral_entropy": _normalized_entropy(excess_analysis),
        "excess_to_baseline_db": _safe_ratio_db(
            float(np.sum(excess_analysis, dtype=np.float64)),
            float(np.sum(baseline_analysis, dtype=np.float64)),
        ),
        "excess_peak_frequency_hz": None,
        "excess_peak_prominence_db": None,
    }
    if float(np.sum(excess_analysis, dtype=np.float64)) > 0:
        peak_index = int(np.argmax(excess_analysis))
        spectral["excess_peak_frequency_hz"] = float(
            analysis_frequencies[peak_index]
        )
        positive = excess_analysis[excess_analysis > 0]
        reference = float(np.median(positive)) if len(positive) else 0.0
        spectral["excess_peak_prominence_db"] = _safe_ratio_db(
            float(excess_analysis[peak_index]),
            reference,
        )

    for quantile in audit["frequency_quantiles"]:
        suffix = int(round(float(quantile) * 100))
        spectral[f"raw_frequency_q{suffix:02d}_hz"] = _frequency_quantile(
            analysis_frequencies,
            raw_analysis,
            float(quantile),
        )
        spectral[f"excess_frequency_q{suffix:02d}_hz"] = _frequency_quantile(
            analysis_frequencies,
            excess_analysis,
            float(quantile),
        )
    for cutoff in audit["retention_cutoffs_hz"]:
        suffix = int(cutoff) // 1000
        spectral[f"raw_energy_below_{suffix}khz_fraction"] = _retention(
            analysis_frequencies,
            raw_analysis,
            float(cutoff),
        )
        spectral[f"excess_energy_below_{suffix}khz_fraction"] = _retention(
            analysis_frequencies,
            excess_analysis,
            float(cutoff),
        )
    tonal_salience = tonal_salience_spectrogram(power, frequencies, audit)
    for prefix in ("core_tonal", "ultrasonic_tonal", "supra96_tonal"):
        spectral.update(
            tonal_component_summary(
                tonal_salience,
                frequencies,
                hop_length / sample_rate_hz,
                audit,
                prefix=prefix,
                lower_hz=float(audit[f"{prefix}_lower_hz"]),
                upper_hz=float(audit[f"{prefix}_upper_hz"]),
            )
        )

    quality_flags = []
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
    if spectral["core_tonal_component_frame_fraction"] < float(
        flags["low_tonal_component_frame_fraction"]
    ):
        quality_flags.append("low_tonal_component_coverage")
    if loop_type is not None:
        quality_flags.append("loop_type_complex_segment")
    elif duration_seconds > float(flags["nonloop_long_duration_seconds"]):
        quality_flags.append("nonloop_duration_gt_3s")

    return {
        "sample_count": sample_count,
        "minimum_pcm16": int(np.min(integer)),
        "maximum_pcm16": int(np.max(integer)),
        "peak_fraction_full_scale": peak_fraction,
        "rms_fraction_full_scale": rms_fraction,
        "rms_dbfs": rms_dbfs,
        "dc_offset_fraction_full_scale": dc_fraction,
        "exact_clipping_fraction": exact_clipping_fraction,
        "near_clipping_fraction": near_clipping_fraction,
        "zero_fraction": zero_fraction,
        "absolute_amplitude_q50": float(amplitude_quantiles[0]),
        "absolute_amplitude_q90": float(amplitude_quantiles[1]),
        "absolute_amplitude_q95": float(amplitude_quantiles[2]),
        "absolute_amplitude_q99": float(amplitude_quantiles[3]),
        "frame_rms_q10": float(frame_rms_quantiles[0]),
        "frame_rms_q50": float(frame_rms_quantiles[1]),
        "frame_rms_q90": float(frame_rms_quantiles[2]),
        "temporal_dynamic_db": temporal_dynamic_db,
        "low_activity_frame_fraction": low_activity_fraction,
        **spectral,
        "quality_flags": quality_flags,
    }


def audit_wav_file(
    path: str | Path,
    audit: Mapping[str, Any],
    flags: Mapping[str, Any],
    *,
    loop_type: int | None,
    duration_seconds: float,
) -> dict[str, Any]:
    sample_rate_hz, samples = wavfile.read(Path(path), mmap=True)
    result = audit_waveform(
        np.asarray(samples),
        int(sample_rate_hz),
        audit,
        flags,
        loop_type=loop_type,
        duration_seconds=duration_seconds,
    )
    result["decoded_sample_rate_hz"] = int(sample_rate_hz)
    return result


def quantile_summary(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    quantiles: Sequence[float] = (0.0, 0.01, 0.05, 0.50, 0.95, 0.99, 1.0),
) -> dict[str, float | int | None]:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return {"n": 0, **{f"q{int(q * 100):02d}": None for q in quantiles}}
    array = np.asarray(values, dtype=np.float64)
    result: dict[str, float | int | None] = {"n": len(values)}
    for q, value in zip(quantiles, np.quantile(array, quantiles), strict=True):
        result[f"q{int(q * 100):02d}"] = float(value)
    return result
