#!/usr/bin/env python3
"""Validate, tabulate and summarize the frozen domain-shift stress results."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_stress_experiments import configured_pairs, scenario_group  # noqa: E402
from dolphin_behavior_mil.experiment import (  # noqa: E402
    atomic_write_json,
    load_toml,
    sha256_file,
)
from dolphin_behavior_mil.modeling_data import LABEL_ORDER  # noqa: E402
from dolphin_behavior_mil.statistical_analysis import probability_metrics  # noqa: E402
from dolphin_behavior_mil.stress_validation import StressScenarioRegistry  # noqa: E402


RESULT_ROOT = PROJECT_ROOT / "results/stress_tests"
SUMMARY_ROOT = RESULT_ROOT / "summary"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError("all CSV rows must have identical ordered fields")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def close_nested(actual: Any, expected: Any, *, tolerance: float = 1e-12) -> bool:
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and set(actual) == set(expected) and all(
            close_nested(actual[key], expected[key], tolerance=tolerance)
            for key in expected
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            close_nested(left, right, tolerance=tolerance)
            for left, right in zip(actual, expected, strict=True)
        )
    if expected is None or isinstance(expected, (str, bool, int)):
        return actual == expected
    if isinstance(expected, float):
        return actual is not None and np.isclose(
            float(actual), expected, rtol=0.0, atol=tolerance, equal_nan=True
        )
    return actual == expected


def main() -> int:
    execution_path = PROJECT_ROOT / "configs/stress_execution_v1.toml"
    execution = load_toml(execution_path)
    registry = StressScenarioRegistry.load(PROJECT_ROOT)
    pairs = configured_pairs(execution, registry)
    main_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    result_hashes: dict[str, str] = {}

    for model_id, scenario_id in pairs:
        selection_path = (
            RESULT_ROOT / "selection" / model_id / scenario_id / "candidate_selection.json"
        )
        result_path = RESULT_ROOT / "test" / model_id / scenario_id / "test_result.json"
        prediction_path = result_path.parent / "test_predictions.jsonl"
        for path in (selection_path, result_path, prediction_path):
            if not path.is_file():
                raise FileNotFoundError(f"incomplete stress pair {model_id}/{scenario_id}: {path}")
        selection = read_json(selection_path)
        result = read_json(result_path)
        rows = sorted(read_jsonl(prediction_path), key=lambda row: row["encounter_id"])
        scenario = registry.scenario(scenario_id)
        expected_ids = tuple(sorted(scenario.test_ids))
        ids = tuple(str(row["encounter_id"]) for row in rows)
        if ids != expected_ids or len(ids) != len(set(ids)):
            raise RuntimeError(f"test coverage mismatch: {model_id}/{scenario_id}")
        if selection.get("outer_test_accessed") is not False:
            raise RuntimeError(f"selection touched outer test: {model_id}/{scenario_id}")
        truth = np.asarray([row["y_true"] for row in rows], dtype=np.int8)
        expected_truth = np.stack([registry.dataset.labels[value] for value in ids])
        if not np.array_equal(truth, expected_truth):
            raise RuntimeError(f"truth mismatch: {model_id}/{scenario_id}")
        raw = np.asarray([row["raw_probability"] for row in rows], dtype=np.float64)
        calibrated = np.asarray(
            [row["calibrated_probability"] for row in rows], dtype=np.float64
        )
        if raw.shape != truth.shape or calibrated.shape != truth.shape:
            raise RuntimeError(f"probability shape mismatch: {model_id}/{scenario_id}")
        if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(calibrated)):
            raise RuntimeError(f"non-finite probability: {model_id}/{scenario_id}")
        if np.any((raw < 0) | (raw > 1)) or np.any((calibrated < 0) | (calibrated > 1)):
            raise RuntimeError(f"out-of-range probability: {model_id}/{scenario_id}")
        raw_metrics = probability_metrics(truth, raw, LABEL_ORDER)
        calibrated_metrics = probability_metrics(truth, calibrated, LABEL_ORDER)
        if not close_nested(result["raw_metrics"], raw_metrics):
            raise RuntimeError(f"raw metric mismatch: {model_id}/{scenario_id}")
        if not close_nested(result["calibrated_metrics"], calibrated_metrics):
            raise RuntimeError(f"calibrated metric mismatch: {model_id}/{scenario_id}")

        main_rows.append(
            {
                "scenario_group": scenario_group(scenario_id),
                "scenario_id": scenario_id,
                "model_id": model_id,
                "train_count": len(scenario.train_ids),
                "fixed_validation_count": len(scenario.validation_ids),
                "test_count": len(scenario.test_ids),
                "buffer_excluded_count": len(scenario.buffer_ids),
                "selected_candidate_id": result["selected_candidate_id"],
                "final_seed_count": len(result["final_seeds"]),
                "raw_macro_ap": raw_metrics["macro_average_precision"],
                "raw_macro_roc_auc": raw_metrics["macro_roc_auc"],
                "raw_macro_brier": raw_metrics["macro_brier_score"],
                "calibrated_macro_ap": calibrated_metrics["macro_average_precision"],
                "calibrated_macro_roc_auc": calibrated_metrics["macro_roc_auc"],
                "calibrated_macro_brier": calibrated_metrics["macro_brier_score"],
                "calibrated_macro_ece": calibrated_metrics["macro_adaptive_ece"],
            }
        )
        for label in LABEL_ORDER:
            roc = raw_metrics["per_label_roc_auc"][label]
            calibrated_roc = calibrated_metrics["per_label_roc_auc"][label]
            label_rows.append(
                {
                    "scenario_group": scenario_group(scenario_id),
                    "scenario_id": scenario_id,
                    "model_id": model_id,
                    "label": label,
                    "positive_count": raw_metrics["positive_counts"][label],
                    "negative_count": raw_metrics["negative_counts"][label],
                    "raw_ap": raw_metrics["per_label_average_precision"][label],
                    "raw_roc_auc": roc["value"],
                    "raw_roc_auc_defined": roc["defined"],
                    "calibrated_ap": calibrated_metrics[
                        "per_label_average_precision"
                    ][label],
                    "calibrated_roc_auc": calibrated_roc["value"],
                    "calibrated_roc_auc_defined": calibrated_roc["defined"],
                    "calibrated_brier": calibrated_metrics["per_label_brier_score"][
                        label
                    ],
                    "calibrated_ece": calibrated_metrics[
                        "per_label_adaptive_ece"
                    ][label],
                }
            )
        result_hashes[str(result_path.relative_to(PROJECT_ROOT))] = sha256_file(
            result_path
        )
        result_hashes[str(prediction_path.relative_to(PROJECT_ROOT))] = sha256_file(
            prediction_path
        )
        result_hashes[str(selection_path.relative_to(PROJECT_ROOT))] = sha256_file(
            selection_path
        )

    by_scenario: dict[str, dict[str, dict[str, Any]]] = {}
    for row in main_rows:
        by_scenario.setdefault(str(row["scenario_id"]), {})[
            str(row["model_id"])
        ] = row
    delta_rows: list[dict[str, Any]] = []
    for scenario_id, values in sorted(by_scenario.items()):
        if "hf_lw_gam" not in values:
            continue
        hf = values["hf_lw_gam"]
        for model_id, row in values.items():
            if model_id == "hf_lw_gam":
                continue
            delta_rows.append(
                {
                    "scenario_group": hf["scenario_group"],
                    "scenario_id": scenario_id,
                    "comparator_model_id": model_id,
                    "hf_lw_gam_raw_macro_ap": hf["raw_macro_ap"],
                    "comparator_raw_macro_ap": row["raw_macro_ap"],
                    "delta_hf_minus_comparator": float(hf["raw_macro_ap"])
                    - float(row["raw_macro_ap"]),
                }
            )

    group_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in main_rows:
        grouped.setdefault(
            (str(row["scenario_group"]), str(row["model_id"])), []
        ).append(float(row["raw_macro_ap"]))
    for (group, model_id), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        group_rows.append(
            {
                "scenario_group": group,
                "model_id": model_id,
                "scenario_count": len(array),
                "median_raw_macro_ap": float(np.median(array)),
                "minimum_raw_macro_ap": float(np.min(array)),
                "maximum_raw_macro_ap": float(np.max(array)),
            }
        )

    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    main_path = SUMMARY_ROOT / "stress_model_scenario_metrics_v1.csv"
    label_path = SUMMARY_ROOT / "stress_per_label_metrics_v1.csv"
    delta_path = SUMMARY_ROOT / "stress_hf_contrasts_v1.csv"
    group_path = SUMMARY_ROOT / "stress_group_descriptives_v1.csv"
    atomic_write_csv(main_path, main_rows)
    atomic_write_csv(label_path, label_rows)
    atomic_write_csv(delta_path, delta_rows)
    atomic_write_csv(group_path, group_rows)

    forward_rows = [
        row for row in main_rows if row["scenario_group"] == "strict_forward"
    ]
    forward_ranking = sorted(
        (
            {"model_id": row["model_id"], "raw_macro_ap": row["raw_macro_ap"]}
            for row in forward_rows
        ),
        key=lambda row: (-float(row["raw_macro_ap"]), str(row["model_id"])),
    )
    for index, row in enumerate(forward_ranking, start=1):
        row["rank"] = index
    summary = {
        "status": "complete",
        "scope": "registered_exploratory_domain_shift_stress_summary",
        "scenario_count": len(registry.scenario_ids),
        "model_scenario_pair_count": len(pairs),
        "model_scenario_rows": len(main_rows),
        "per_label_rows": len(label_rows),
        "hf_contrast_rows": len(delta_rows),
        "strict_forward_test_count": len(
            registry.scenario(
                "forward_train_2021_2023_validate_2024_test_2025"
            ).test_ids
        ),
        "strict_forward_raw_macro_ap_ranking": forward_ranking,
        "interpretation": (
            "All stress results are exploratory diagnostics from the same "
            "observational programme; none is independent external validation "
            "or evidence of per-whistle behavior truth or real-time deployment."
        ),
        "source_sha256": {
            str(execution_path.relative_to(PROJECT_ROOT)): sha256_file(execution_path),
            "scripts/run_stress_experiments.py": sha256_file(
                PROJECT_ROOT / "scripts/run_stress_experiments.py"
            ),
            "src/dolphin_behavior_mil/stress_validation.py": sha256_file(
                PROJECT_ROOT / "src/dolphin_behavior_mil/stress_validation.py"
            ),
            **result_hashes,
        },
    }
    summary_path = SUMMARY_ROOT / "stress_summary_v1.json"
    atomic_write_json(summary_path, summary)
    summary["artifact_sha256"] = {
        str(path.relative_to(SUMMARY_ROOT)): sha256_file(path)
        for path in (main_path, label_path, delta_path, group_path)
    }
    atomic_write_json(summary_path, summary)
    print(
        f"complete: {len(main_rows)} registered pairs, "
        f"{len(label_rows)} per-label rows",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
