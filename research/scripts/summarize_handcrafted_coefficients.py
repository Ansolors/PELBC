#!/usr/bin/env python3
"""Post hoc descriptive stability summary for locked handcrafted baselines."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dolphin_behavior_mil.experiment import atomic_write_json, sha256_file  # noqa: E402
from dolphin_behavior_mil.modeling_data import LABEL_ORDER  # noqa: E402


MODEL_ID = "handcrafted_logistic"
OUTER_FOLDS = tuple(f"outer_{index:02d}" for index in range(1, 6))
RESULT_ROOT = PROJECT_ROOT / "results/posthoc_handcrafted_coefficients"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError("CSV rows have inconsistent fields")
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


def official_run(outer_fold_id: str) -> Path:
    result_path = (
        PROJECT_ROOT
        / "results/nested_cv/outer_folds"
        / MODEL_ID
        / outer_fold_id
        / "outer_fold_result.json"
    )
    result = read_json(result_path)
    candidate_id = str(result["selected_candidate_id"])
    seed = int(result["final_seeds"][0])
    pattern = (
        f"outer_final__{MODEL_ID}__{outer_fold_id}__no_inner__"
        f"{candidate_id}__{seed}__*"
    )
    matches = []
    for path in (PROJECT_ROOT / "results/model_runs").glob(pattern):
        manifest_path = path / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path)
        if manifest.get("status") != "completed":
            continue
        identity = manifest.get("identity", {})
        if (
            identity.get("model_id") == MODEL_ID
            and identity.get("outer_fold_id") == outer_fold_id
            and identity.get("candidate_id") == candidate_id
            and int(identity.get("seed")) == seed
        ):
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"expected one official run for {outer_fold_id}, found {matches}")
    return matches[0]


def main() -> int:
    names_reference: list[str] | None = None
    coefficients = []
    c_values = []
    run_records = []
    source_hashes: dict[str, str] = {}

    for outer_fold_id in OUTER_FOLDS:
        run_dir = official_run(outer_fold_id)
        manifest_path = run_dir / "run_manifest.json"
        normalizer_path = run_dir / "normalizer.json"
        checkpoint_path = run_dir / "best_checkpoint.pt"
        manifest = read_json(manifest_path)
        normalizer = read_json(normalizer_path)
        for artifact in ("normalizer.json", "best_checkpoint.pt"):
            actual = sha256_file(run_dir / artifact)
            if actual != manifest["artifact_sha256"][artifact]:
                raise RuntimeError(f"artifact hash mismatch: {run_dir.name}/{artifact}")
        if normalizer["training_partition_sha256"] != manifest["train_partition_sha256"]:
            raise RuntimeError(f"normalizer partition mismatch: {outer_fold_id}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["model_id"] != MODEL_ID or tuple(checkpoint["label_order"]) != LABEL_ORDER:
            raise RuntimeError(f"checkpoint identity mismatch: {outer_fold_id}")
        names = list(normalizer["output_feature_names"])
        if names_reference is None:
            names_reference = names
        elif names != names_reference:
            raise RuntimeError("fold-local output feature names differ")
        state = checkpoint["model_state"]
        heads = state["heads"]
        if len(heads) != len(LABEL_ORDER) or any(
            head.get("kind") != "logistic_regression" for head in heads
        ):
            raise RuntimeError(f"unexpected classical head: {outer_fold_id}")
        matrix = np.asarray([head["coefficient"][0] for head in heads], dtype=np.float64)
        if matrix.shape != (len(LABEL_ORDER), len(names)) or not np.all(np.isfinite(matrix)):
            raise RuntimeError(f"invalid coefficient matrix: {outer_fold_id}")
        coefficients.append(matrix)
        c_values.append(float(state["c_value"]))
        run_records.append(
            {
                "outer_fold_id": outer_fold_id,
                "run_id": run_dir.name,
                "candidate_id": checkpoint["candidate"]["candidate_id"],
                "c_value": float(state["c_value"]),
                "train_partition_sha256": manifest["train_partition_sha256"],
            }
        )
        for path in (manifest_path, normalizer_path, checkpoint_path):
            source_hashes[str(path.relative_to(PROJECT_ROOT))] = sha256_file(path)

    assert names_reference is not None
    names = names_reference
    raw = np.stack(coefficients, axis=0)
    norms = np.linalg.norm(raw, axis=2, keepdims=True)
    if np.any(norms <= 0):
        raise RuntimeError("a coefficient vector has zero L2 norm")
    normalized = raw / norms
    all_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    correlations: dict[str, Any] = {}

    for label_index, label in enumerate(LABEL_ORDER):
        pairwise = []
        for left in range(len(OUTER_FOLDS)):
            for right in range(left + 1, len(OUTER_FOLDS)):
                value = float(spearmanr(raw[left, label_index], raw[right, label_index]).statistic)
                if not np.isfinite(value):
                    raise RuntimeError(f"undefined rank correlation: {label}")
                pairwise.append(value)
        correlations[label] = {
            "pair_count": len(pairwise),
            "median_spearman": float(np.median(pairwise)),
            "minimum_spearman": float(np.min(pairwise)),
            "maximum_spearman": float(np.max(pairwise)),
            "values": pairwise,
        }

        candidates = []
        for feature_index, feature in enumerate(names):
            values = raw[:, label_index, feature_index]
            scaled = normalized[:, label_index, feature_index]
            positive = int(np.sum(values > 0))
            negative = int(np.sum(values < 0))
            zero = int(np.sum(values == 0))
            nonzero = positive + negative
            consistency = float(max(positive, negative) / nonzero) if nonzero else 0.0
            direction = "positive" if positive > negative else "negative" if negative > positive else "mixed"
            family, _, aggregation = feature.partition("__")
            row = {
                "label": label,
                "feature": feature,
                "measurement_family": family,
                "aggregation": aggregation,
                "positive_folds": positive,
                "negative_folds": negative,
                "zero_folds": zero,
                "direction_consistency_fraction": consistency,
                "common_direction": direction,
                "median_coefficient": float(np.median(values)),
                "median_absolute_coefficient": float(np.median(np.abs(values))),
                "median_absolute_l2_normalized_coefficient": float(np.median(np.abs(scaled))),
                **{
                    f"{outer_fold_id}_coefficient": float(values[index])
                    for index, outer_fold_id in enumerate(OUTER_FOLDS)
                },
            }
            all_rows.append(row)
            if nonzero == len(OUTER_FOLDS) and consistency == 1.0:
                candidates.append(row)

        family_best: dict[str, dict[str, Any]] = {}
        for row in candidates:
            family = str(row["measurement_family"])
            previous = family_best.get(family)
            if previous is None or float(row["median_absolute_l2_normalized_coefficient"]) > float(
                previous["median_absolute_l2_normalized_coefficient"]
            ):
                family_best[family] = row
        ranked = sorted(
            family_best.values(),
            key=lambda row: (
                -float(row["median_absolute_l2_normalized_coefficient"]),
                str(row["feature"]),
            ),
        )
        for rank, row in enumerate(ranked[:8], start=1):
            top_rows.append({"rank_within_label": rank, **row})

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    all_path = RESULT_ROOT / "all_feature_coefficients_v1.csv"
    top_path = RESULT_ROOT / "top_stable_measurement_families_v1.csv"
    summary_path = RESULT_ROOT / "handcrafted_coefficient_stability_v1.json"
    atomic_write_csv(all_path, all_rows)
    atomic_write_csv(top_path, top_rows)
    atomic_write_json(
        summary_path,
        {
            "status": "complete",
            "scope": "posthoc_descriptive_locked_training_parameter_inspection",
            "model_id": MODEL_ID,
            "outer_fold_count": len(OUTER_FOLDS),
            "feature_count": len(names),
            "label_order": list(LABEL_ORDER),
            "test_labels_used_for_feature_ranking": False,
            "primary_endpoint_or_model_selection_changed": False,
            "coefficient_scaling_for_ranking": "within_fold_label_l2_normalization",
            "selected_c_values": c_values,
            "runs": run_records,
            "pairwise_fold_rank_correlations": correlations,
            "top_stable_measurement_families": top_rows,
            "interpretation_guard": (
                "Outer training pools overlap and selected regularization differs. "
                "Correlated coefficients are descriptive model parameters, not independent "
                "replicates, causal markers, significance tests, or whistle-level labels."
            ),
            "artifact_sha256": {
                all_path.name: sha256_file(all_path),
                top_path.name: sha256_file(top_path),
            },
            "source_sha256": {
                "scripts/summarize_handcrafted_coefficients.py": sha256_file(Path(__file__)),
                **source_hashes,
            },
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "folds": len(OUTER_FOLDS),
                "features": len(names),
                "top_rows": len(top_rows),
                "summary": str(summary_path.relative_to(PROJECT_ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
