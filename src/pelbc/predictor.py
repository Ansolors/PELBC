"""High-level encounter-directory inference API."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
import hashlib
import math
from pathlib import Path
from typing import Any

from .audio import analyze_wav
from .errors import InputValidationError
from .features import aggregate_whistles
from .model import PortableAcousticLR
from .version import __version__


ProgressCallback = Callable[[int, int, Path], None]


def _discover_wavs(directory: Path, recursive: bool) -> list[Path]:
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    files = [
        path
        for path in iterator
        if path.is_file() and path.suffix.casefold() == ".wav"
    ]
    return sorted(files, key=lambda path: path.relative_to(directory).as_posix().casefold())


def _input_set_sha256(clip_details: list[dict[str, Any]]) -> str:
    canonical = "".join(
        f"{row['relative_path']}\t{row['sha256']}\n" for row in clip_details
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class EncounterPredictor:
    """Score one encounter represented by a variable-size bag of whistle WAVs."""

    def __init__(self, model: PortableAcousticLR | None = None) -> None:
        self.model = model or PortableAcousticLR.load()

    @classmethod
    def from_model_path(cls, path: str | Path) -> "EncounterPredictor":
        return cls(PortableAcousticLR.load(path))

    def predict_directory(
        self,
        input_dir: str | Path,
        *,
        encounter_id: str | None = None,
        recursive: bool = False,
        include_clip_details: bool = True,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        directory = Path(input_dir).expanduser().resolve()
        if not directory.is_dir():
            raise InputValidationError(f"input directory does not exist: {directory}")
        paths = _discover_wavs(directory, recursive)
        if not paths:
            scope = "recursively" if recursive else "at the directory top level"
            raise InputValidationError(f"no .wav files found {scope}: {directory}")

        extraction = self.model.record["feature_extraction"]
        audit = extraction["audit"]
        flags = extraction["descriptive_flags"]
        analyses: list[dict[str, Any]] = []
        for index, path in enumerate(paths, start=1):
            if progress is not None:
                progress(index, len(paths), path)
            analyses.append(analyze_wav(path, audit, flags))

        aggregate_record = aggregate_whistles(
            [row["features"] for row in analyses],
            per_whistle_features=tuple(extraction["per_whistle_features"]),
            aggregation_statistics=tuple(extraction["aggregation_statistics"]),
            add_missing_fraction_per_feature=bool(
                extraction["add_missing_fraction_per_feature"]
            ),
        )
        scores, standardized = self.model.predict(aggregate_record)

        clip_details = []
        for path, analysis in zip(paths, analyses, strict=True):
            features = analysis["features"]
            clip_details.append(
                {
                    "relative_path": path.relative_to(directory).as_posix(),
                    "sha256": analysis["sha256"],
                    "sample_rate_hz": analysis["sample_rate_hz"],
                    "sample_count": analysis["sample_count"],
                    "duration_seconds": analysis["duration_seconds"],
                    "core_tonal_component_found": bool(
                        features["core_tonal_component_found"]
                    ),
                    "quality_flags": list(analysis["quality_flags"]),
                }
            )

        warnings: list[dict[str, Any]] = []

        def add_warning(
            code: str,
            message: str,
            *,
            affected_files: list[str] | None = None,
            details: Any | None = None,
        ) -> None:
            record: dict[str, Any] = {"code": code, "message": message}
            if affected_files:
                record["affected_files"] = affected_files
            if details is not None:
                record["details"] = details
            warnings.append(record)

        if len(paths) < 3:
            add_warning(
                "sparse_whistle_evidence",
                "Fewer than three whistle clips were supplied; treat score differences cautiously.",
            )

        training_rates = {
            int(value) for value in self.model.record["training"]["native_sample_rates_hz"]
        }
        observed_rates = {int(row["sample_rate_hz"]) for row in analyses}
        unobserved_rates = sorted(observed_rates - training_rates)
        if unobserved_rates:
            add_warning(
                "sample_rate_outside_training_regimes",
                "At least one sample rate was not represented in the model-development cohort.",
                details={
                    "observed_unseen_hz": unobserved_rates,
                    "training_hz": sorted(training_rates),
                },
            )

        tonal_missing = [
            row["relative_path"]
            for row in clip_details
            if not row["core_tonal_component_found"]
        ]
        if tonal_missing:
            add_warning(
                "core_tonal_component_not_found",
                "The deterministic 2--40 kHz tonal-component audit did not find a component in some clips.",
                affected_files=tonal_missing,
            )

        files_by_flag: dict[str, list[str]] = defaultdict(list)
        for row in clip_details:
            for flag in row["quality_flags"]:
                files_by_flag[str(flag)].append(str(row["relative_path"]))
        for flag, affected in sorted(files_by_flag.items()):
            add_warning(
                f"audio_quality:{flag}",
                f"The descriptive audio-quality flag '{flag}' was raised.",
                affected_files=affected,
            )

        hashes = Counter(str(row["sha256"]) for row in clip_details)
        duplicated_hashes = sorted(value for value, count in hashes.items() if count > 1)
        if duplicated_hashes:
            duplicate_files = [
                str(row["relative_path"])
                for row in clip_details
                if row["sha256"] in duplicated_hashes
            ]
            add_warning(
                "duplicate_audio_content",
                "Two or more input paths contain byte-identical audio and may overweight a whistle.",
                affected_files=duplicate_files,
            )

        extreme = sorted(
            (
                {
                    "feature": name,
                    "standardized_value": float(value),
                }
                for name, value in standardized.items()
                if math.isfinite(value) and abs(value) > 6.0
            ),
            key=lambda row: abs(row["standardized_value"]),
            reverse=True,
        )
        if extreme:
            add_warning(
                "aggregate_feature_extrapolation",
                "Some encounter aggregates are more than six training standard deviations from their means.",
                details=extreme[:10],
            )

        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], self.model.label_order.index(item[0])),
        )
        ranked_behaviors = [
            {"rank": rank, "label": label_name, "score": float(score)}
            for rank, (label_name, score) in enumerate(ranked, start=1)
        ]
        total_duration = float(sum(row["duration_seconds"] for row in analyses))
        resolved_encounter_id = str(encounter_id or directory.name).strip()
        if not resolved_encounter_id:
            raise InputValidationError("encounter ID cannot be empty")

        result: dict[str, Any] = {
            "schema_version": "pelbc_prediction_v1",
            "tool": {
                "name": "PELBC",
                "version": __version__,
            },
            "task": "encounter_level_multilabel_behavioral_context_scoring",
            "encounter_id": resolved_encounter_id,
            "input_summary": {
                "whistle_clip_count": len(paths),
                "total_clip_duration_seconds": total_duration,
                "sample_rates_hz": sorted(observed_rates),
                "input_set_sha256": _input_set_sha256(clip_details),
            },
            "model": {
                "model_id": self.model.record["model_id"],
                "model_type": self.model.record["model_type"],
                "artifact_sha256": self.model.artifact_sha256,
                "training_encounter_count": int(
                    self.model.record["training"]["encounter_count"]
                ),
                "score_type": self.model.record["score_semantics"]["type"],
            },
            "behavior_scores": {label: float(scores[label]) for label in self.model.label_order},
            "ranked_behaviors": ranked_behaviors,
            "binary_decisions": None,
            "quality_control": {
                "status": "warning" if warnings else "ok",
                "warning_count": len(warnings),
                "warnings": warnings,
            },
            "interpretation": {
                "scope": "Scores describe the behavioral context of the encounter bag, not individual whistles or callers.",
                "multilabel": "Each behavior is scored independently; several contexts may co-occur.",
                "calibration": "Scores are not prospectively calibrated probabilities for a new survey.",
                "thresholds": "No universal binary threshold is supplied; operating thresholds require survey-specific validation.",
                "deployment": "This offline back end assumes whistles have already been detected and segmented.",
            },
        }
        if include_clip_details:
            result["clip_details"] = clip_details
        return result
