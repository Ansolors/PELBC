"""Frozen waveform resampling and log-mel extraction for whistle instances."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import math
from typing import Any, Mapping

import numpy as np
from scipy import fft
from scipy.signal import resample_poly, windows


PCM16_SCALE = 32768.0


def pcm16_to_float32(samples: np.ndarray) -> np.ndarray:
    if samples.dtype != np.int16:
        raise ValueError(f"expected int16 PCM, got {samples.dtype}")
    if samples.ndim != 1:
        raise ValueError("expected mono waveform")
    return samples.astype(np.float32) / PCM16_SCALE


def remove_dc_mean(samples: np.ndarray) -> np.ndarray:
    if samples.ndim != 1:
        raise ValueError("expected mono waveform")
    return samples.astype(np.float32, copy=False) - np.mean(samples, dtype=np.float64)


def resample_waveform(
    samples: np.ndarray,
    original_sample_rate_hz: int,
    target_sample_rate_hz: int,
    *,
    kaiser_beta: float = 5.0,
    padtype: str = "constant",
) -> np.ndarray:
    if min(original_sample_rate_hz, target_sample_rate_hz) <= 0:
        raise ValueError("sample rates must be positive")
    if samples.ndim != 1:
        raise ValueError("expected mono waveform")
    samples = samples.astype(np.float32, copy=False)
    if original_sample_rate_hz == target_sample_rate_hz:
        return samples.copy()
    divisor = math.gcd(original_sample_rate_hz, target_sample_rate_hz)
    up = target_sample_rate_hz // divisor
    down = original_sample_rate_hz // divisor
    output = resample_poly(
        samples,
        up,
        down,
        window=("kaiser", float(kaiser_beta)),
        padtype=padtype,
    )
    return np.asarray(output, dtype=np.float32)


def hz_to_mel_htk(frequency_hz: np.ndarray | float) -> np.ndarray:
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    return 2595.0 * np.log10(1.0 + frequency / 700.0)


def mel_to_hz_htk(mel: np.ndarray | float) -> np.ndarray:
    mel = np.asarray(mel, dtype=np.float64)
    return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)


@lru_cache(maxsize=32)
def mel_filterbank(
    sample_rate_hz: int,
    n_fft: int,
    n_mels: int,
    f_min_hz: float,
    f_max_hz: float,
) -> np.ndarray:
    """Construct an HTK triangular mel bank with Slaney area normalization."""

    nyquist = sample_rate_hz / 2.0
    if not 0 <= f_min_hz < f_max_hz <= nyquist:
        raise ValueError("mel frequency limits must lie within Nyquist")
    if min(n_fft, n_mels) <= 0:
        raise ValueError("n_fft and n_mels must be positive")
    mel_edges = np.linspace(
        hz_to_mel_htk(f_min_hz),
        hz_to_mel_htk(f_max_hz),
        n_mels + 2,
    )
    hz_edges = mel_to_hz_htk(mel_edges)
    frequencies = fft.rfftfreq(n_fft, d=1.0 / sample_rate_hz)
    filters = np.zeros((n_mels, len(frequencies)), dtype=np.float64)
    for index in range(n_mels):
        lower, center, upper = hz_edges[index : index + 3]
        left = (frequencies - lower) / max(center - lower, np.finfo(float).eps)
        right = (upper - frequencies) / max(upper - center, np.finfo(float).eps)
        filters[index] = np.maximum(0.0, np.minimum(left, right))
        filters[index] *= 2.0 / max(upper - lower, np.finfo(float).eps)
    return filters.astype(np.float32)


def _right_padded_frames(
    samples: np.ndarray,
    win_length: int,
    hop_length: int,
) -> np.ndarray:
    if min(win_length, hop_length) <= 0:
        raise ValueError("window and hop must be positive")
    if len(samples) <= win_length:
        frame_count = 1
    else:
        frame_count = 1 + math.ceil((len(samples) - win_length) / hop_length)
    target_length = (frame_count - 1) * hop_length + win_length
    if len(samples) < target_length:
        samples = np.pad(samples, (0, target_length - len(samples)), mode="constant")
    view = np.lib.stride_tricks.sliding_window_view(samples, win_length)
    return np.asarray(view[::hop_length][:frame_count], dtype=np.float32)


def logmel_spectrogram(
    waveform: np.ndarray,
    *,
    sample_rate_hz: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    n_mels: int,
    f_min_hz: float,
    f_max_hz: float,
    top_db: float,
) -> np.ndarray:
    if n_fft < win_length:
        raise ValueError("n_fft must be at least win_length")
    frames = _right_padded_frames(
        waveform.astype(np.float32, copy=False),
        win_length,
        hop_length,
    )
    window = windows.hann(win_length, sym=False).astype(np.float32)
    spectrum = fft.rfft(frames * window[None, :], n=n_fft, axis=1, workers=1)
    power = np.square(np.abs(spectrum), dtype=np.float32)
    power /= max(float(np.sum(np.square(window), dtype=np.float64)), np.finfo(float).eps)
    filters = mel_filterbank(sample_rate_hz, n_fft, n_mels, f_min_hz, f_max_hz)
    mel_power = filters @ power.T
    floor = np.finfo(np.float32).tiny
    log_power = 10.0 * np.log10(np.maximum(mel_power, floor))
    reference = float(np.max(log_power))
    relative = log_power - reference
    relative = np.maximum(relative, -float(top_db))
    return np.asarray(relative, dtype=np.float32)


def extract_frozen_logmel(
    pcm16_samples: np.ndarray,
    original_sample_rate_hz: int,
    config: Mapping[str, Mapping[str, Any]],
    *,
    bandwidth_sensitivity: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return resampled waveform and frozen relative-dB log-mel feature."""

    waveform = pcm16_to_float32(pcm16_samples)
    if bool(config["preprocessing"]["remove_per_clip_dc_mean"]):
        waveform = remove_dc_mean(waveform)
    if bandwidth_sensitivity:
        target_rate = int(config["bandwidth_sensitivity"]["sample_rate_hz"])
        n_fft = int(config["bandwidth_sensitivity"]["n_fft"])
        win_length = int(config["bandwidth_sensitivity"]["win_length"])
        hop_length = int(config["bandwidth_sensitivity"]["hop_length"])
        n_mels = int(config["bandwidth_sensitivity"]["n_mels"])
        f_min = float(config["bandwidth_sensitivity"]["f_min_hz"])
        f_max = float(config["bandwidth_sensitivity"]["f_max_hz"])
    else:
        target_rate = int(config["preprocessing"]["target_sample_rate_hz"])
        n_fft = int(config["stft"]["n_fft"])
        win_length = int(config["stft"]["win_length"])
        hop_length = int(config["stft"]["hop_length"])
        n_mels = int(config["mel"]["n_mels"])
        f_min = float(config["mel"]["f_min_hz"])
        f_max = float(config["mel"]["f_max_hz"])
    resampled = resample_waveform(
        waveform,
        original_sample_rate_hz,
        target_rate,
        kaiser_beta=float(config["resampling"]["kaiser_beta"]),
        padtype=str(config["resampling"]["padtype"]),
    )
    feature = logmel_spectrogram(
        resampled,
        sample_rate_hz=target_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min_hz=f_min,
        f_max_hz=f_max,
        top_db=float(config["log"]["top_db"]),
    )
    return resampled, feature


def feature_sha256(feature: np.ndarray, *, dtype: str = "float16") -> str:
    canonical = np.asarray(feature, dtype=np.dtype(dtype), order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()
