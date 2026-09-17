#!/usr/bin/env python3
"""Export a portable full-cohort Acoustic LR model from the research repository.

This maintainer utility is not needed for ordinary inference. It requires the
private/derived research project and scikit-learn, then emits a dependency-light
JSON artifact consumed by :mod:`pelbc.model`.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pelbc.model import MODEL_SCHEMA  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def selected_outer_candidates(results_root: Path) -> list[dict[str, Any]]:
    rows = []
    selection_root = results_root / "nested_cv" / "selection" / "handcrafted_logistic"
    for path in sorted(selection_root.glob("outer_*/candidate_selection.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        candidate = record["selected_candidate"]["candidate"]
        rows.append(
            {
                "outer_fold_id": record["outer_fold_id"],
                "candidate_id": candidate["candidate_id"],
                "c_value": float(candidate["c_value"]),
                "class_weight": candidate["class_weight"],
            }
        )
    if len(rows) != 5:
        raise RuntimeError(f"expected five outer selections, found {len(rows)}")
    return rows


def build_model_record(project_root: Path) -> dict[str, Any]:
    experiment_root = (
        project_root if (project_root / "configs").is_dir()
        else project_root / "experiments"
    )
    research_source = experiment_root / "src"
    if str(research_source) not in sys.path:
        sys.path.insert(0, str(research_source))

    from dolphin_behavior_mil.classical import (  # noqa: PLC0415
        FoldLocalTabularEncoder,
        IndependentLogisticBaseline,
        classical_candidate_grid,
        handcrafted_records,
        label_matrix,
    )
    from dolphin_behavior_mil.modeling_data import DatasetRegistry  # noqa: PLC0415

    modeling_path = experiment_root / "configs" / "modeling_protocol_v1.toml"
    audit_path = experiment_root / "configs" / "audio_audit_v1.toml"
    modeling = read_toml(modeling_path)
    audit_config = read_toml(audit_path)
    registry = DatasetRegistry.load(
        experiment_root,
        dataset_version="0.4.0",
        validate_cache_files=False,
    )
    encounter_ids = registry.encounter_ids
    section = modeling["handcrafted_logistic"]
    records = handcrafted_records(
        registry,
        encounter_ids,
        per_whistle_features=tuple(section["per_whistle_features"]),
        aggregation_statistics=tuple(section["aggregation_statistics"]),
        add_missing_fraction_per_feature=bool(
            section["add_missing_fraction_per_feature"]
        ),
    )
    numeric_names = tuple(sorted(set(records[0]) - {"encounter_id"}))
    encoder = FoldLocalTabularEncoder.fit(
        records,
        numeric_features=numeric_names,
        categorical_features=(),
    )
    matrix = encoder.fit_transform(records)
    labels = label_matrix(registry, encounter_ids)

    outer_candidates = selected_outer_candidates(experiment_root / "results")
    counts = Counter(row["candidate_id"] for row in outer_candidates)
    modal_id, modal_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    candidate_grid = classical_candidate_grid(
        modeling["classical"]["search"]["c_values"],
        modeling["classical"]["search"]["class_weight_options"],
    )
    candidates = {row["candidate_id"]: row for row in candidate_grid}
    candidate = candidates[modal_id]
    classifier = IndependentLogisticBaseline(
        c_value=float(candidate["c_value"]),
        class_weight=candidate["class_weight"],
        random_seed=20260829,
        maximum_iterations=int(modeling["classical"]["common"]["maximum_iterations"]),
    ).fit(matrix, labels)

    scores = classifier.predict_proba(matrix)
    if scores.shape != (234, 4) or not np.all(np.isfinite(scores)):
        raise RuntimeError("full-cohort classifier produced invalid training scores")

    native_sample_rates = sorted(
        {
            int(whistle["sample_rate_hz"])
            for encounter_id in encounter_ids
            for whistle in registry.whistles[encounter_id]
        }
    )
    whistle_count = sum(len(registry.whistles[value]) for value in encounter_ids)
    bootstrap_path = (
        experiment_root / "results" / "statistical_analysis" / "bootstrap_intervals_v1.json"
    )
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    interval = bootstrap["intervals"]["handcrafted_logistic"]["raw"][
        "macro_average_precision"
    ]

    source_paths = {
        "modeling_protocol": modeling_path,
        "audio_audit_protocol": audit_path,
        "encounters": registry.version_dir / "encounters.jsonl",
        "labels": registry.version_dir / "labels.jsonl",
        "whistles": registry.version_dir / "whistles.jsonl",
        "outer_assignments": registry.version_dir / "outer_fold_assignments.jsonl",
        "bootstrap_intervals": bootstrap_path,
    }
    return {
        "schema_version": MODEL_SCHEMA,
        "model_id": "acoustic_lr_full_234_v1",
        "model_type": "independent_l2_logistic_heads_on_encounter_acoustic_aggregates",
        "label_order": list(registry.label_order),
        "score_semantics": {
            "type": "uncalibrated_full_data_logistic_score",
            "range": [0.0, 1.0],
            "independent_multilabel_heads": True,
            "binary_thresholds_included": False,
            "interpretation": "Relative encounter-level behavioral-context scores; not prospectively calibrated probabilities.",
        },
        "feature_extraction": {
            "implementation": "native_audio_quality_spectral_v1_compatible_subset",
            "input": "pre-extracted_mono_pcm16_whistle_wav",
            "per_whistle_features": list(section["per_whistle_features"]),
            "aggregation_statistics": list(section["aggregation_statistics"]),
            "add_missing_fraction_per_feature": bool(
                section["add_missing_fraction_per_feature"]
            ),
            "audit": audit_config["audit"],
            "descriptive_flags": audit_config["descriptive_flags"],
        },
        "encoder": encoder.to_dict(),
        "classifier": classifier.to_dict(),
        "training": {
            "dataset_version": "0.4.0",
            "encounter_count": len(encounter_ids),
            "whistle_count": whistle_count,
            "native_sample_rates_hz": native_sample_rates,
            "training_partition_sha256": encoder.training_partition_sha256,
            "label_prevalence": {
                label_name: float(np.mean(labels[:, index]))
                for index, label_name in enumerate(registry.label_order)
            },
            "final_fit_candidate": dict(candidate),
            "final_fit_candidate_rule": "modal_candidate_id_across_five_locked_outer_fold_selections",
            "modal_selection_count": modal_count,
            "outer_fold_selected_candidates": outer_candidates,
            "random_seed": 20260829,
        },
        "performance_reference": {
            "scope": "algorithm-level pooled out-of-fold nested-cross-validation estimate; not an in-sample estimate for this full-data refit",
            "metric": "raw_macro_average_precision",
            "point_estimate": float(interval["point_estimate"]),
            "campaign_cluster_bootstrap_95_percent_ci": [
                float(interval["lower"]),
                float(interval["upper"]),
            ],
            "campaign_count": int(bootstrap["campaign_count"]),
        },
        "provenance": {
            "source_sha256": {
                name: sha256_file(path) for name, path in source_paths.items()
            },
            "exporter": "tools/export_acoustic_lr.py",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--research-project",
        required=True,
        type=Path,
        help="path to the local research directory containing configs, private data and results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "private_models" / "acoustic_lr.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.research_project.expanduser().resolve()
    if not ((project_root / "configs").is_dir() or (project_root / "experiments/configs").is_dir()):
        raise SystemExit(f"not a research project: {project_root}")
    record = build_model_record(project_root)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(sha256_file(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
