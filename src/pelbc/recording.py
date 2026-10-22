"""Recording-to-context inference with automatic whistle-segment extraction."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Callable
import wave

from .errors import InputValidationError, PELBCError
from .predictor import EncounterPredictor, _discover_wavs
from .version import __version__


ProgressCallback = Callable[[int, int, str], None]
MODEL_DIRECTORY = Path(__file__).resolve().parent / "models"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probe(path: Path) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as reader:
            if reader.getnchannels() != 1 or reader.getsampwidth() != 2:
                raise InputValidationError(f"录音必须为单声道 PCM16 WAV：{path}")
            if reader.getcomptype() != "NONE" or reader.getframerate() < 80_000:
                raise InputValidationError(f"录音必须为未压缩 WAV，采样率不低于 80 kHz：{path}")
            if reader.getnframes() == 0:
                raise InputValidationError(f"录音为空：{path}")
            return {
                "path": str(path),
                "sample_rate_hz": reader.getframerate(),
                "sample_count": reader.getnframes(),
                "duration_seconds": reader.getnframes() / reader.getframerate(),
                "sha256": _sha256(path),
            }
    except (wave.Error, EOFError) as exc:
        raise InputValidationError(f"无法读取 PCM WAV：{path}: {exc}") from exc


def export_pcm_clip(source: Path, target: Path, start_s: float, end_s: float) -> dict[str, Any]:
    """Export floor/ceil sample boundaries while preserving the native PCM bytes."""
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        first = max(0, min(params.nframes, math.floor(start_s * params.framerate)))
        stop = max(first, min(params.nframes, math.ceil(end_s * params.framerate)))
        if stop <= first:
            raise InputValidationError("检测事件对应的音频片段为空。")
        reader.setpos(first)
        frames = reader.readframes(stop - first)
    with wave.open(str(target), "wb") as writer:
        writer.setnchannels(params.nchannels)
        writer.setsampwidth(params.sampwidth)
        writer.setframerate(params.framerate)
        writer.setcomptype(params.comptype, params.compname)
        writer.writeframes(frames)
    return {
        "clip_file": target.name,
        "start_sample": first,
        "end_sample_exclusive": stop,
        "start_seconds": first / params.framerate,
        "end_seconds": stop / params.framerate,
        "sha256": _sha256(target),
    }


class RecordingPredictor:
    """Automatically extract whistle segments and score one encounter.

    A directory may contain several source recordings, all belonging to the same
    encounter. Each run has a fresh clip directory so stale files cannot enter a bag.
    """

    def __init__(
        self,
        predictor: EncounterPredictor | None = None,
        *,
        detector_weights: str | Path | None = None,
        confidence: float = 0.62,
        clip_padding_s: float = 0.25,
        device: str = "auto",
    ) -> None:
        if not math.isfinite(confidence) or not 0 < confidence <= 1:
            raise InputValidationError("检测置信度必须位于 (0, 1]。")
        if not math.isfinite(clip_padding_s) or clip_padding_s < 0:
            raise InputValidationError("片段前后保留时长必须为非负有限数。")
        self.predictor = predictor or EncounterPredictor()
        self.weights = (
            Path(detector_weights).expanduser().resolve()
            if detector_weights is not None
            else MODEL_DIRECTORY / "whistle_segment_yolo.pt"
        )
        if not self.weights.is_file():
            raise InputValidationError(
                f"找不到 whistle 检测模型：{self.weights}。"
                "请通过 --detector-model 或 detector_weights 提供本地检测权重。"
            )
        self.weight_hash = _sha256(self.weights)
        if detector_weights is None:
            metadata = json.loads(
                (MODEL_DIRECTORY / "whistle_segment_yolo_metadata.json").read_text(encoding="utf-8")
            )
            if self.weight_hash != metadata["sha256"]:
                raise InputValidationError("内置 whistle 检测模型校验失败，请重新安装软件。")
        self.confidence = float(confidence)
        self.clip_padding_s = float(clip_padding_s)
        self.device = device

    def _detector(self):
        try:
            from .detector import DetectorConfig, WhistleDetector
        except ImportError as exc:
            raise PELBCError("自动提取依赖未安装；请在软件目录运行 python -m pip install .") from exc
        config = DetectorConfig(confidence=self.confidence)
        return WhistleDetector(
            self.weights, config, device=None if self.device == "auto" else self.device
        )

    def predict(
        self,
        source: str | Path,
        *,
        work_dir: str | Path | None = None,
        encounter_id: str | None = None,
        recursive: bool = False,
        include_clip_details: bool = True,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        source_path = Path(source).expanduser().resolve()
        if source_path.is_file() and source_path.suffix.casefold() == ".wav":
            paths = [source_path]
            default_id = source_path.stem
        elif source_path.is_dir():
            paths = _discover_wavs(source_path, recursive)
            default_id = source_path.name
        else:
            raise InputValidationError(f"输入必须为 WAV 录音或录音文件夹：{source_path}")
        if not paths:
            raise InputValidationError(f"没有找到 WAV 录音：{source_path}")
        identifier = str(encounter_id or default_id).strip()
        if not identifier:
            raise InputValidationError("遭遇编号不能为空。")
        destination = (
            Path(work_dir).expanduser().resolve()
            if work_dir is not None
            else source_path.parent / f"{default_id}_pelbc"
        )
        if source_path.is_dir() and (
            destination == source_path or source_path in destination.parents
        ):
            raise InputValidationError("输出文件夹必须位于原始录音文件夹之外，避免片段混入后续输入。")

        # Validate every input before inference or creating an output bag.
        sources = [_probe(path) for path in paths]
        detector = self._detector()
        destination.mkdir(parents=True, exist_ok=True)
        run_directory = Path(tempfile.mkdtemp(prefix="run_", dir=destination))
        clip_directory = run_directory / "whistles"
        clip_directory.mkdir()
        manifest_path = run_directory / "extraction_manifest.json"
        manifest: dict[str, Any] = {
            "schema_version": "pelbc_extraction_v1",
            "software": {"name": "PELBC", "version": __version__},
            "encounter_id": identifier,
            "status": "running",
            "method": "PELBC automatic whistle-segment extraction",
            "detector": {
                "checkpoint": self.weights.name,
                "sha256": self.weight_hash,
                "config": asdict(detector.config),
                "device": str(detector.device),
            },
            "clip_padding_seconds": self.clip_padding_s,
            "sources": sources,
            "events": [],
        }

        def save_manifest() -> None:
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

        save_manifest()
        try:
            for source_index, (path, source_record) in enumerate(zip(paths, sources), start=1):
                def detector_progress(done: int, total: int, message: str) -> None:
                    if progress is not None:
                        progress(done, total, f"录音 {source_index}/{len(paths)} · {path.name} · {message}")

                events = detector.detect(path, progress=detector_progress)
                for event in events:
                    clip_path = clip_directory / f"R{source_index:04d}_{event.event_id}.wav"
                    audio = export_pcm_clip(
                        path,
                        clip_path,
                        max(0.0, event.start_s - self.clip_padding_s),
                        min(source_record["duration_seconds"], event.end_s + self.clip_padding_s),
                    )
                    manifest["events"].append(
                        {"source_index": source_index, "detection": event.as_dict(), "audio": audio}
                    )
                save_manifest()
            clip_count = len(manifest["events"])
            manifest["status"] = "complete" if clip_count else "no_whistles_detected"
            manifest["whistle_clip_count"] = clip_count
            save_manifest()
            if clip_count:
                def feature_progress(index: int, total: int, path: Path) -> None:
                    if progress is not None:
                        progress(index, total, f"行为预测特征 {index}/{total} · {path.name}")

                result = self.predictor.predict_directory(
                    clip_directory,
                    encounter_id=identifier,
                    include_clip_details=include_clip_details,
                    progress=feature_progress,
                )
                result["status"] = "complete"
            else:
                result = self._empty_result(identifier)
            result["schema_version"] = "pelbc_recording_prediction_v1"
            result["input_summary"].update({
                "input_type": "original_encounter_recordings",
                "recording_count": len(paths),
                "total_recording_duration_seconds": sum(row["duration_seconds"] for row in sources),
            })
            result["extraction"] = {
                "manifest_path": str(manifest_path),
                "clip_directory": str(clip_directory),
                "detector_sha256": self.weight_hash,
                "confidence": self.confidence,
                "clip_padding_seconds": self.clip_padding_s,
                "whistle_clip_count": clip_count,
            }
            result["interpretation"]["deployment"] = (
                "PELBC automatically extracted whistle segments from the supplied recordings "
                "and used the resulting encounter bag for behavioral-context prediction."
            )
            (run_directory / "prediction.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return result
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["error"] = str(exc)
            save_manifest()
            if isinstance(exc, ImportError):
                raise PELBCError("自动提取依赖未安装；请运行 python -m pip install .") from exc
            raise

    def _empty_result(self, identifier: str) -> dict[str, Any]:
        model = self.predictor.model
        return {
            "software": {"name": "PELBC", "version": __version__},
            "task": "encounter_level_multilabel_behavioral_context_scoring",
            "encounter_id": identifier,
            "status": "no_whistles_detected",
            "input_summary": {"whistle_clip_count": 0, "total_clip_duration_seconds": 0.0},
            "model": {"model_id": model.record["model_id"], "artifact_sha256": model.artifact_sha256},
            "behavior_scores": None,
            "ranked_behaviors": [],
            "binary_decisions": None,
            "quality_control": {
                "status": "insufficient_evidence",
                "warning_count": 1,
                "warnings": [{
                    "code": "no_whistles_detected",
                    "message": "未检出 whistle，未计算行为分数；这不表示四类行为均未发生。",
                }],
            },
            "interpretation": {
                "scope": "The prediction unit is the encounter, not an individual animal.",
                "calibration": "No behavioral probability is available without an extracted whistle bag.",
            },
        }
