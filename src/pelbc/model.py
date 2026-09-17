"""Portable JSON model loading and dependency-light logistic inference."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ModelValidationError
from .features import AGGREGATION_STATISTICS, PER_WHISTLE_FEATURES, encode_aggregates


MODEL_SCHEMA = "pelbc_acoustic_lr_model_v1"


def _sigmoid(value: float) -> float:
    if value >= 0:
        return float(1.0 / (1.0 + math.exp(-value)))
    exponential = math.exp(value)
    return float(exponential / (1.0 + exponential))


@dataclass(frozen=True)
class PortableAcousticLR:
    record: dict[str, Any]
    artifact_sha256: str
    source: str

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PortableAcousticLR":
        if path is None:
            raise ModelValidationError(
                "This source-only distribution contains no trained model. "
                "Supply a private local model with --model or from_model_path()."
            )
        try:
            model_path = Path(path).expanduser().resolve()
            payload = model_path.read_bytes()
            source = str(model_path)
            record = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ModelValidationError(f"cannot load model artifact: {exc}") from exc
        if not isinstance(record, dict):
            raise ModelValidationError("model artifact must contain one JSON object")
        model = cls(
            record=record,
            artifact_sha256=hashlib.sha256(payload).hexdigest(),
            source=source,
        )
        model.validate()
        return model

    @property
    def label_order(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.record["label_order"])

    def validate(self) -> None:
        if self.record.get("schema_version") != MODEL_SCHEMA:
            raise ModelValidationError(
                f"unsupported model schema {self.record.get('schema_version')!r}"
            )
        if self.label_order != ("Feeding", "Travelling", "Milling", "Socializing"):
            raise ModelValidationError("model label order differs from the frozen task")
        extraction = self.record.get("feature_extraction")
        if not isinstance(extraction, dict):
            raise ModelValidationError("model is missing feature-extraction settings")
        if tuple(extraction.get("per_whistle_features", ())) != PER_WHISTLE_FEATURES:
            raise ModelValidationError("model requests an unsupported whistle feature contract")
        if tuple(extraction.get("aggregation_statistics", ())) != AGGREGATION_STATISTICS:
            raise ModelValidationError("model requests an unsupported aggregation contract")
        if extraction.get("add_missing_fraction_per_feature") is not True:
            raise ModelValidationError("model must include per-feature missing fractions")
        if not isinstance(extraction.get("audit"), dict) or not isinstance(
            extraction.get("descriptive_flags"), dict
        ):
            raise ModelValidationError("model is missing acoustic audit settings")
        encoder = self.record.get("encoder")
        classifier = self.record.get("classifier")
        if not isinstance(encoder, dict) or not isinstance(classifier, dict):
            raise ModelValidationError("model is missing encoder or classifier state")
        feature_count = len(encoder.get("output_feature_names", []))
        heads = classifier.get("heads")
        if not isinstance(heads, list) or len(heads) != len(self.label_order):
            raise ModelValidationError("classifier must contain four independent heads")
        for label_name, head in zip(self.label_order, heads, strict=True):
            if not isinstance(head, dict) or head.get("kind") != "logistic_regression":
                raise ModelValidationError(f"invalid classifier head for {label_name}")
            coefficient = np.asarray(head.get("coefficient"), dtype=np.float64)
            intercept = np.asarray(head.get("intercept"), dtype=np.float64)
            if (
                coefficient.shape != (1, feature_count)
                or intercept.shape != (1,)
                or not np.all(np.isfinite(coefficient))
                or not np.all(np.isfinite(intercept))
            ):
                raise ModelValidationError(f"invalid coefficient state for {label_name}")

    def predict(
        self,
        aggregate_record: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        try:
            vector, standardized = encode_aggregates(
                aggregate_record,
                self.record["encoder"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelValidationError(f"cannot apply model encoder: {exc}") from exc

        scores: dict[str, float] = {}
        for label_name, head in zip(
            self.label_order,
            self.record["classifier"]["heads"],
            strict=True,
        ):
            coefficient = np.asarray(head["coefficient"], dtype=np.float64)[0]
            intercept = float(head["intercept"][0])
            logit = float(np.dot(coefficient, vector) + intercept)
            scores[label_name] = _sigmoid(logit)
        return scores, standardized
