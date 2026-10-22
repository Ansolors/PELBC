"""面向长录音的滑窗 YOLO whistle 检测器。

这个模块只负责把 WAV 映射为全局时间-频率事件：

    WAV -> 3 s 重叠窗 -> 0--30 kHz 声谱图 -> YOLO 多框
        -> 绝对坐标 -> 跨窗加权合并

导出 WAV 时的前后冗余不写入检测框，由上层 pipeline 统一处理。
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from PIL import Image
from scipy import signal


ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class DetectorConfig:
    window_s: float = 3.0
    stride_s: float = 1.0
    target_sample_rate_hz: int = 96_000
    fmax_hz: float = 30_000.0
    window_ms: float = 20.0
    hop_ms: float = 2.5
    dynamic_range_db: float = 70.0
    vmax_percentile: float = 99.7
    image_width: int = 512
    image_height: int = 256
    inference_imgsz: int = 512
    confidence: float = 0.62
    nms_iou: float = 0.70
    max_detections_per_window: int = 300
    batch_size: int = 8
    merge_time_iou: float = 0.25
    merge_temporal_overlap: float = 0.55
    merge_frequency_iou: float = 0.10
    merge_max_gap_s: float = 0.04

    def validate(self) -> None:
        if self.window_s <= 0 or self.stride_s <= 0:
            raise ValueError("window_s 和 stride_s 必须大于 0")
        if self.stride_s > self.window_s:
            raise ValueError("stride_s 不能大于 window_s")
        if self.target_sample_rate_hz < 2 * self.fmax_hz:
            raise ValueError("目标采样率必须高于分析频带 Nyquist 下限")
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("confidence 必须位于 (0, 1]")
        if self.batch_size <= 0 or self.max_detections_per_window <= 0:
            raise ValueError("batch_size 和 max_detections_per_window 必须大于 0")


@dataclass
class DetectionEvent:
    source_wav: str
    event_id: str
    start_s: float
    end_s: float
    fmin_hz: float
    fmax_hz: float
    confidence: float
    contributing_windows: list[int] = field(default_factory=list)
    raw_detection_count: int = 1

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WindowInfo:
    index: int
    start_sample: int
    valid_samples: int
    sample_rate_hz: int
    spectrogram_time_start_s: float
    spectrogram_time_end_s: float
    frequency_min_hz: float
    frequency_max_hz: float


def choose_device() -> str | int:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return "cpu"


def _window_starts(total_samples: int, window_samples: int, hop_samples: int) -> list[int]:
    """在原始样点域生成确定性起点，并在必要时添加尾对齐窗。

    这与训练工程 ``audio.iter_window_starts`` 保持同一语义，避免先在
    秒域用浮点 ``arange`` 再四舍五入造成尾窗偏移或重复。
    """

    if total_samples <= 0 or window_samples <= 0 or hop_samples <= 0:
        raise ValueError("音频长度、窗口长度和步长必须为正样点数")
    if total_samples <= window_samples:
        return [0]
    starts = list(range(0, total_samples - window_samples + 1, hop_samples))
    tail = total_samples - window_samples
    if starts[-1] != tail:
        starts.append(tail)
    return starts


def _pcm_to_float(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.unsignedinteger):
        info = np.iinfo(data.dtype)
        midpoint = (float(info.max) + float(info.min)) / 2.0
        scale = max((float(info.max) - float(info.min)) / 2.0, 1.0)
        return ((data.astype(np.float32) - midpoint) / scale).astype(np.float32)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = float(max(abs(info.min), abs(info.max), 1))
        return (data.astype(np.float32) / scale).astype(np.float32)
    return np.nan_to_num(data.astype(np.float32), copy=False)


def _resample_mono(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1, dtype=np.float32)
    if source_rate == target_rate:
        return waveform
    divisor = math.gcd(int(source_rate), int(target_rate))
    up = int(target_rate // divisor)
    down = int(source_rate // divisor)
    return signal.resample_poly(waveform, up, down, window=("kaiser", 8.6)).astype(
        np.float32, copy=False
    )


def waveform_to_yolo_image(
    waveform: np.ndarray,
    sample_rate_hz: int,
    cfg: DetectorConfig,
) -> tuple[np.ndarray, float, float, float, float]:
    """返回 RGB 声谱图以及图像对应的时间/频率边界。"""

    cfg.validate()
    # 固定窗必须在源采样率、原始 PCM 域先补零/截断。正式训练数据与
    # whistle_detector.audio.fixed_window_pcm 采用这一顺序；若先去均值或
    # 重采样再补零，短 WAV 的声谱图会与训练分布不一致。
    source = np.asarray(waveform)
    if source.ndim not in (1, 2):
        raise ValueError(f"不支持的 WAV 数据形状：{source.shape}")
    required_source = max(1, int(round(cfg.window_s * int(sample_rate_hz))))
    if source.shape[0] < required_source:
        pad_shape = (required_source - source.shape[0], *source.shape[1:])
        source = np.concatenate((source, np.zeros(pad_shape, dtype=source.dtype)), axis=0)
    elif source.shape[0] > required_source:
        source = source[:required_source]

    x = _resample_mono(_pcm_to_float(source), sample_rate_hz, cfg.target_sample_rate_hz)
    x = np.nan_to_num(x, copy=False)
    if x.size == 0:
        x = np.zeros(1, dtype=np.float32)
    x = x - np.float32(np.mean(x, dtype=np.float64))
    required = int(round(cfg.window_s * cfg.target_sample_rate_hz))
    if x.size < required:
        x = np.pad(x, (0, required - x.size), mode="constant")
    elif x.size > required:
        x = x[:required]

    nperseg = max(16, int(round(cfg.window_ms * cfg.target_sample_rate_hz / 1000.0)))
    hop = max(1, int(round(cfg.hop_ms * cfg.target_sample_rate_hz / 1000.0)))
    nfft = 1 << (nperseg - 1).bit_length()
    frequencies, times, spectrum = signal.stft(
        x,
        fs=cfg.target_sample_rate_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg - hop,
        nfft=nfft,
        detrend=False,
        return_onesided=True,
        boundary=None,
        padded=True,
        scaling="spectrum",
    )
    keep = frequencies <= cfg.fmax_hz
    frequencies = frequencies[keep]
    power = np.abs(spectrum[keep]) ** 2
    raw_db = 10.0 * np.log10(np.maximum(power, np.finfo(np.float32).tiny))
    finite = raw_db[np.isfinite(raw_db)]
    vmax = float(np.percentile(finite, cfg.vmax_percentile)) if finite.size else 0.0
    vmin = vmax - cfg.dynamic_range_db
    normalized = np.clip((raw_db - vmin) / max(vmax - vmin, 1e-7), 0.0, 1.0)

    from matplotlib import colormaps

    rgb = colormaps["magma"](np.flipud(normalized), bytes=True)[..., :3]
    image = Image.fromarray(rgb, mode="RGB").resize(
        (cfg.image_width, cfg.image_height), Image.Resampling.BILINEAR
    )
    time_step = hop / cfg.target_sample_rate_hz
    frequency_step = cfg.target_sample_rate_hz / nfft
    # PNG 像素代表 STFT bin 的面积，坐标映射应用 bin 边界而非
    # bin 中心。这与连续模型训练数据构建器的定义完全一致。
    return (
        np.asarray(image, dtype=np.uint8),
        float(times[0] - time_step / 2.0),
        float(times[-1] + time_step / 2.0),
        max(0.0, float(frequencies[0] - frequency_step / 2.0)),
        min(cfg.target_sample_rate_hz / 2.0, float(frequencies[-1] + frequency_step / 2.0)),
    )


def _normalized_box_to_event(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    info: WindowInfo,
    total_samples: int,
) -> tuple[int, int, float, float]:
    """Convert one normalized YOLO box to the training pipeline's sample semantics."""

    if total_samples <= 0:
        raise ValueError("音频总样点数必须大于 0")
    x0 = max(0.0, float(x_center) - float(width) / 2.0)
    x1 = min(1.0, float(x_center) + float(width) / 2.0)
    y0 = max(0.0, float(y_center) - float(height) / 2.0)
    y1 = min(1.0, float(y_center) + float(height) / 2.0)
    time_span = max(
        1e-9,
        info.spectrogram_time_end_s - info.spectrogram_time_start_s,
    )
    frequency_span = max(
        1e-9,
        info.frequency_max_hz - info.frequency_min_hz,
    )
    valid_duration_s = info.valid_samples / info.sample_rate_hz
    local_start_s = info.spectrogram_time_start_s + x0 * time_span
    local_end_s = info.spectrogram_time_start_s + x1 * time_span
    local_start_s = float(np.clip(local_start_s, 0.0, valid_duration_s))
    local_end_s = float(np.clip(local_end_s, 0.0, valid_duration_s))

    event_start = info.start_sample + int(round(local_start_s * info.sample_rate_hz))
    event_end = info.start_sample + int(round(local_end_s * info.sample_rate_hz))
    event_start = max(0, min(total_samples - 1, event_start))
    # 与训练工程正式推理一致：坐标四舍五入后即使两端落在
    # 同一样点，也保留一个样点，而不是丢弃小于 10 ms 的检测框。
    event_end = max(event_start + 1, min(total_samples, event_end))
    fmax = info.frequency_max_hz - y0 * frequency_span
    fmin = info.frequency_max_hz - y1 * frequency_span
    return event_start, event_end, float(fmin), float(fmax)


def _interval_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    intersection = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return intersection / union if union > 0 else 0.0


def _overlap_coefficient(a0: float, a1: float, b0: float, b1: float) -> float:
    intersection = max(0.0, min(a1, b1) - max(a0, b0))
    denominator = min(max(0.0, a1 - a0), max(0.0, b1 - b0))
    return intersection / denominator if denominator > 0 else 0.0


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    ordered = sorted(zip(values, weights), key=lambda pair: pair[0])
    halfway = sum(max(0.0, weight) for _, weight in ordered) / 2.0
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += max(0.0, weight)
        if cumulative >= halfway:
            return float(value)
    return float(ordered[-1][0])


def _cluster_bounds(cluster: Sequence[DetectionEvent]) -> tuple[float, float, float, float]:
    weights = [max(1e-6, item.confidence) for item in cluster]
    return (
        min(item.start_s for item in cluster),
        max(item.end_s for item in cluster),
        _weighted_median([item.fmin_hz for item in cluster], weights),
        _weighted_median([item.fmax_hz for item in cluster], weights),
    )


def _association_score(
    item: DetectionEvent,
    cluster: Sequence[DetectionEvent],
    cfg: DetectorConfig,
) -> float | None:
    item_windows = set(item.contributing_windows)
    cluster_windows = {
        index for existing in cluster for index in existing.contributing_windows
    }
    if item_windows & cluster_windows:
        return None
    start, end, fmin, fmax = _cluster_bounds(cluster)
    temporal_iou = _interval_iou(item.start_s, item.end_s, start, end)
    temporal_overlap = _overlap_coefficient(item.start_s, item.end_s, start, end)
    frequency_iou = _interval_iou(item.fmin_hz, item.fmax_hz, fmin, fmax)
    gap = max(0.0, max(start, item.start_s) - min(end, item.end_s))
    temporal_ok = (
        temporal_iou >= cfg.merge_time_iou
        or temporal_overlap >= cfg.merge_temporal_overlap
        or gap <= cfg.merge_max_gap_s
    )
    if not temporal_ok or frequency_iou < cfg.merge_frequency_iou:
        return None
    return 2.0 * temporal_iou + temporal_overlap + frequency_iou - gap


def merge_detections(
    detections: Sequence[DetectionEvent], cfg: DetectorConfig
) -> list[DetectionEvent]:
    """对跨重叠窗检测做置信度加权框融合。"""

    ordered = sorted(detections, key=lambda item: (item.start_s, -item.confidence))
    clusters: list[list[DetectionEvent]] = []
    for item in ordered:
        candidates: list[tuple[float, int]] = []
        for index, cluster in enumerate(clusters):
            score = _association_score(item, cluster, cfg)
            if score is not None:
                candidates.append((score, index))
        if not candidates:
            clusters.append([item])
        else:
            _, target_index = max(candidates)
            clusters[target_index].append(item)

    merged: list[DetectionEvent] = []
    for cluster in clusters:
        windows = sorted({index for item in cluster for index in item.contributing_windows})
        start, end, fmin, fmax = _cluster_bounds(cluster)
        merged.append(
            DetectionEvent(
                source_wav=cluster[0].source_wav,
                event_id="",
                start_s=start,
                end_s=end,
                fmin_hz=fmin,
                fmax_hz=fmax,
                confidence=float(max(item.confidence for item in cluster)),
                contributing_windows=windows,
                raw_detection_count=sum(item.raw_detection_count for item in cluster),
            )
        )
    # 正式推理只按事件起点稳定排序；同起点时保留聚类创建顺序。
    merged.sort(key=lambda item: item.start_s)
    for index, item in enumerate(merged, start=1):
        item.event_id = f"W{index:04d}"
    return merged


class WhistleDetector:
    def __init__(
        self,
        weights: str | Path,
        config: DetectorConfig | None = None,
        device: str | int | None = None,
    ) -> None:
        self.weights = Path(weights).expanduser().resolve()
        if not self.weights.is_file():
            raise FileNotFoundError(f"找不到 YOLO 权重：{self.weights}")
        self.config = config or DetectorConfig()
        self.config.validate()
        self.device = choose_device() if device is None else device
        self._model = None

    def _load_model(self):
        if self._model is None:
            # 权重是本项目自行训练的可信文件。PyTorch>=2.6 默认
            # weights_only=True，而 Ultralytics checkpoint 还包含模型结构。
            os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
            from ultralytics import YOLO

            self._model = YOLO(str(self.weights))
        return self._model

    def detect(
        self,
        wav_path: str | Path,
        progress: ProgressCallback | None = None,
    ) -> list[DetectionEvent]:
        from scipy.io import wavfile

        source = Path(wav_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"找不到录音：{source}")
        sample_rate, pcm = wavfile.read(source, mmap=True)
        if pcm.ndim not in (1, 2):
            raise ValueError(f"不支持的 WAV 数据形状：{pcm.shape}")
        total_samples = int(pcm.shape[0])
        if total_samples <= 0:
            raise ValueError(f"WAV 为空：{source}")
        window_samples = int(round(self.config.window_s * sample_rate))
        hop_samples = int(round(self.config.stride_s * sample_rate))
        starts = _window_starts(total_samples, window_samples, hop_samples)
        model = self._load_model()
        raw: list[DetectionEvent] = []

        for batch_start in range(0, len(starts), self.config.batch_size):
            selected = starts[batch_start : batch_start + self.config.batch_size]
            images: list[np.ndarray] = []
            infos: list[WindowInfo] = []
            for offset, start_sample in enumerate(selected):
                end_sample = min(
                    total_samples,
                    start_sample + window_samples,
                )
                chunk = np.asarray(pcm[start_sample:end_sample])
                rgb, t0, t1, f0, f1 = waveform_to_yolo_image(
                    chunk, int(sample_rate), self.config
                )
                # Ultralytics 把 numpy 图像解释为 OpenCV BGR。
                images.append(np.ascontiguousarray(rgb[..., ::-1]))
                infos.append(
                    WindowInfo(
                        index=batch_start + offset,
                        start_sample=start_sample,
                        valid_samples=end_sample - start_sample,
                        sample_rate_hz=int(sample_rate),
                        spectrogram_time_start_s=t0,
                        spectrogram_time_end_s=t1,
                        frequency_min_hz=f0,
                        frequency_max_hz=f1,
                    )
                )
            predictions = model.predict(
                source=images,
                conf=self.config.confidence,
                iou=self.config.nms_iou,
                max_det=self.config.max_detections_per_window,
                imgsz=self.config.inference_imgsz,
                batch=len(images),
                device=self.device,
                save=False,
                verbose=False,
            )
            if len(predictions) != len(images):
                raise RuntimeError("模型返回的窗口数与输入不一致")
            for info, result in zip(infos, predictions):
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                boxes = result.boxes.xywhn.detach().cpu().tolist()
                confidences = result.boxes.conf.detach().cpu().tolist()
                for (x, y, width, height), confidence in zip(boxes, confidences):
                    start_sample, end_sample, fmin, fmax = _normalized_box_to_event(
                        float(x),
                        float(y),
                        float(width),
                        float(height),
                        info,
                        total_samples,
                    )
                    raw.append(
                        DetectionEvent(
                            source_wav=str(source),
                            event_id="",
                            start_s=start_sample / sample_rate,
                            end_s=end_sample / sample_rate,
                            fmin_hz=float(max(0.0, fmin)),
                            fmax_hz=float(min(self.config.fmax_hz, fmax)),
                            confidence=float(confidence),
                            contributing_windows=[info.index],
                        )
                    )
            if progress is not None:
                completed = min(batch_start + len(selected), len(starts))
                progress(completed, len(starts), f"检测声谱窗 {completed}/{len(starts)}")
        return merge_detections(raw, self.config)


__all__ = [
    "DetectionEvent",
    "DetectorConfig",
    "WhistleDetector",
    "choose_device",
    "merge_detections",
    "waveform_to_yolo_image",
]
