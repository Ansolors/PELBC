#!/usr/bin/env python3
"""Small, explicitly post hoc diagnostics; never overwrite frozen experiments.

Run from experiments/: python3 scripts/run_posthoc_revision_diagnostics.py
All new feature definitions, comparisons and sensitivity settings are written
before fitting. Bootstrap intervals condition on fitted OOF predictions; they
are pointwise descriptive intervals, not new confirmatory tests.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import platform
import sys
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dolphin_behavior_mil.classical import (  # noqa: E402
    FoldLocalTabularEncoder, IndependentLogisticBaseline, bag_size_records,
    classical_candidate_grid, handcrafted_records, metadata_records,
)
from dolphin_behavior_mil.experiment import (  # noqa: E402
    atomic_write_json, atomic_write_jsonl, load_toml, sha256_file,
)
from dolphin_behavior_mil.modeling_data import DatasetRegistry, LABEL_ORDER  # noqa: E402
from dolphin_behavior_mil.splitting import campaign_ids_by_date  # noqa: E402
from dolphin_behavior_mil.statistical_analysis import draw_valid_campaign_bootstrap_plan  # noqa: E402

OUT = ROOT / "results/posthoc_revision_diagnostics"
SEED = 20260917
FOLDS = tuple(f"outer_{i:02d}" for i in range(1, 6))
INNER = tuple(f"inner_{i:02d}" for i in range(1, 5))
STRUCTURE = (
    "duration_seconds", "core_tonal_component_frequency_q05_hz",
    "core_tonal_component_frequency_q50_hz", "core_tonal_component_frequency_q95_hz",
    "core_tonal_component_frequency_max_hz", "core_tonal_component_bandwidth_q90_hz",
    "core_tonal_component_duration_seconds",
)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric(y, p):
    if np.any(y.sum(axis=0) == 0) or np.any(y.sum(axis=0) == len(y)):
        raise ValueError("Both classes are required for every reported label")
    return float(average_precision_score(y, p, average="macro"))


def bootstrap_weights(ids, group_by_id, *, primary=False):
    groups = tuple(sorted({group_by_id[i] for i in ids}))
    if primary:
        with np.load(ROOT / "results/statistical_analysis/bootstrap_ap_replicates_v1.npz") as z:
            assert tuple(z["campaign_order"].tolist()) == groups
            draws = z["campaign_draws"].copy()
    else:
        plan = draw_valid_campaign_bootstrap_plan(
            np.stack([REG.labels[i] for i in ids]), [group_by_id[i] for i in ids],
            resamples=10000, seed=SEED,
        )
        assert plan.campaign_order == groups
        draws = plan.draws
    counts = np.zeros((len(draws), len(groups)), dtype=np.int16)
    np.add.at(counts, (np.arange(len(draws))[:, None], draws), 1)
    by_group = {g: k for k, g in enumerate(groups)}
    weights = counts[:, [by_group[group_by_id[i]] for i in ids]]
    return groups, draws, weights


def weighted_bootstrap_ap(y, p, weights):
    """Exact repeated-observation AP using campaign multiplicities, with ties.

    Vectorization only changes computation. Independently checked below against
    sklearn on explicitly repeated encounter rows, including tied predictions.
    """
    result = np.empty((len(weights), y.shape[1]))
    for label in range(y.shape[1]):
        order = np.argsort(-p[:, label], kind="stable")
        scores = p[order, label]
        ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
        w = weights[:, order].astype(float)
        positive = np.cumsum(w * y[order, label][None, :], axis=1)[:, ends]
        total = np.cumsum(w, axis=1)[:, ends]
        precision = np.divide(positive, total, out=np.zeros_like(positive), where=total > 0)
        increments = np.diff(positive, prepend=np.zeros((len(weights), 1)), axis=1)
        result[:, label] = np.sum(precision * increments, axis=1) / positive[:, -1]
    for b in np.linspace(0, len(weights) - 1, 20, dtype=int):
        repeated = np.repeat(np.arange(len(y)), weights[b])
        expected = average_precision_score(y[repeated], p[repeated], average=None)
        np.testing.assert_allclose(result[b], expected, rtol=1e-12, atol=1e-12)
    return result


def interval(values):
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)


def load_official(model):
    rows = sorted(read_rows(ROOT / f"results/nested_cv/oof/{model}/oof_predictions.jsonl"),
                  key=lambda row: row["encounter_id"])
    assert tuple(row["encounter_id"] for row in rows) == REG.encounter_ids
    for row in rows:
        assert row["campaign_id"] == REG.campaign_by_encounter[row["encounter_id"]]
        np.testing.assert_array_equal(row["y_true"], REG.labels[row["encounter_id"]])
    return np.asarray([row["raw_probability"] for row in rows])


def fit_predict(model_id, train_ids, test_ids, candidate):
    y = np.stack([REG.labels[i] for i in train_ids])
    if model_id == "prior_constant":
        return np.tile(y.mean(axis=0), (len(test_ids), 1)), None, None
    records, numeric, categorical = FEATURES[model_id]
    train_records = [records[i] for i in train_ids]
    encoder = FoldLocalTabularEncoder.fit(train_records, numeric_features=numeric,
                                         categorical_features=categorical)
    model = IndependentLogisticBaseline(c_value=candidate["c_value"],
        class_weight=candidate["class_weight"], random_seed=SEED)
    model.fit(encoder.transform(train_records), y)
    predictions = model.predict_proba(encoder.transform([records[i] for i in test_ids]))
    return predictions, encoder.to_dict(), model.to_dict()


def check_partition(train, test, groups):
    assert train and test and not (set(train) & set(test))
    assert not ({groups[i] for i in train} & {groups[i] for i in test})
    metric(np.stack([REG.labels[i] for i in train]), np.full((len(train), 4), 0.5))
    metric(np.stack([REG.labels[i] for i in test]), np.full((len(test), 4), 0.5))


def run_nested(scenario, model_id, ids, groups):
    allowed = set(ids)
    predictions, selections, splits = {}, [], []
    for fold in FOLDS:
        original_train, original_test = REG.outer_split(fold)
        test = tuple(i for i in original_test if i in allowed)
        test_groups = {groups[i] for i in test}
        train = tuple(i for i in original_train if i in allowed and groups[i] not in test_groups)
        check_partition(train, test, groups)
        trials = []
        candidates = [None] if model_id == "prior_constant" else CANDIDATES
        inner_splits = []
        for inner in INNER:
            inner_train, inner_valid = REG.inner_split(fold, inner)
            valid = tuple(i for i in inner_valid if i in train)
            valid_groups = {groups[i] for i in valid}
            fit = tuple(i for i in inner_train if i in train and groups[i] not in valid_groups)
            check_partition(fit, valid, groups)
            inner_splits.append((inner, fit, valid))
        for candidate in candidates:
            scores = []
            for _, fit, valid in inner_splits:
                p, _, _ = fit_predict(model_id, fit, valid, candidate)
                scores.append(metric(np.stack([REG.labels[i] for i in valid]), p))
            trials.append({"candidate": candidate, "fold_macro_ap": scores,
                           "mean_macro_ap": float(np.mean(scores))})
        best = max(row["mean_macro_ap"] for row in trials)
        # Stable tie rule declared before fitting; no run-time latency tie break.
        eligible = [row for row in trials if best - row["mean_macro_ap"] <= 0.001 + 1e-15]
        selected = min(eligible, key=lambda row: (row["candidate"] or {}).get("candidate_id", ""))
        p, encoder, model = fit_predict(model_id, train, test, selected["candidate"])
        predictions.update({i: value for i, value in zip(test, p)})
        selections.append({"outer_fold": fold, "selected_candidate": selected["candidate"],
                           "trials": trials, "encoder": encoder, "model": model})
        splits.append({"outer_fold": fold, "train": train, "test": test,
            "excluded_from_original_training": sorted(set(original_train) - set(train)),
            "inner": [{"fold": k, "train": fit, "validation": valid}
                      for k, fit, valid in inner_splits]})
    assert set(predictions) == allowed
    dest = OUT / scenario / model_id
    dest.mkdir(parents=True, exist_ok=True)
    atomic_write_json(dest / "selection_and_models.json", selections)
    atomic_write_json(dest / "partitions.json", splits)
    atomic_write_jsonl(dest / "oof_predictions.jsonl", [
        {"encounter_id": i, "campaign_id": groups[i],
         "outer_fold_id": REG.outer_fold_by_encounter[i], "y_true": REG.labels[i].astype(int).tolist(),
         "label_order": LABEL_ORDER, "raw_probability": predictions[i].tolist()}
        for i in ids])
    return np.stack([predictions[i] for i in ids])


def prepare_features():
    config = load_toml(ROOT / "configs/modeling_protocol_v1.toml")
    all_ids = REG.encounter_ids
    acoustic = handcrafted_records(REG, all_ids,
        per_whistle_features=config["handcrafted_logistic"]["per_whistle_features"],
        aggregation_statistics=config["handcrafted_logistic"]["aggregation_statistics"],
        add_missing_fraction_per_feature=True)
    numeric = tuple(k for k in acoustic[0] if k != "encounter_id")
    meta_config = config["metadata_logistic"]
    meta = metadata_records(REG, all_ids, **{k: meta_config[k] for k in (
        "numeric_features", "categorical_features", "forbidden_source_prefixes", "forbidden_source_fields")})
    bags = bag_size_records(REG, all_ids)
    acoustic_map = {row["encounter_id"]: row for row in acoustic}
    meta_map = {row["encounter_id"]: row for row in meta}
    structure_names = tuple(k for k in numeric if k.split("__")[0] in STRUCTURE)
    quality_names = tuple(k for k in numeric if k not in structure_names)
    assert len(structure_names) == len(quality_names) == 56
    features = {
        "handcrafted_logistic": (acoustic_map, numeric, ()),
        "structure_logistic": (acoustic_map, structure_names, ()),
        "quality_logistic": (acoustic_map, quality_names, ()),
        "metadata_logistic": (meta_map, tuple(meta_config["numeric_features"]), tuple(meta_config["categorical_features"])),
        "bag_size_logistic": ({row["encounter_id"]: row for row in bags}, tuple(config["bag_size_logistic"]["feature_names"]), ()),
        "metadata_acoustic_logistic": ({i: {**meta_map[i], **acoustic_map[i]} for i in all_ids},
            tuple(meta_config["numeric_features"]) + numeric, tuple(meta_config["categorical_features"])),
    }
    candidates = classical_candidate_grid(config["classical"]["search"]["c_values"],
                                          config["classical"]["search"]["class_weight_options"])
    return config, features, candidates


def dataset_audit():
    rows = read_rows(REG.version_dir / "labels.jsonl")
    states = {state: sum(r["behavior_sum_vs_number_of_scans"] == state for r in rows)
              for state in sorted({r["behavior_sum_vs_number_of_scans"] for r in rows})}
    hashes = defaultdict(list)
    for i in REG.encounter_ids:
        for whistle in REG.whistles[i]:
            hashes[whistle["clip_sha256"]].append((i, whistle["whistle_id"]))
    cross = [v for v in hashes.values() if len({i for i, _ in v}) > 1]
    audit = {"primary_encounters": len(rows), "count_consistency": states,
        "modeled_clip_count": sum(len(v) for v in REG.whistles.values()),
        "unique_modeled_clip_file_hashes": len(hashes),
        "cross_encounter_identical_file_hash_groups": cross,
        "scope": "Existing SHA-256 file hashes; does not establish absence of near-duplicates or shared callers.",
        "all_four_primary_labels_zero": [{"encounter_id": r["encounter_id"], "behavior_counts": r["behavior_counts"]}
                                         for r in rows if not any(r["primary_label_vector"])]}
    atomic_write_json(OUT / "dataset_audit.json", audit)
    return audit


def main():
    global REG, FEATURES, CANDIDATES
    warnings.simplefilter("error", ConvergenceWarning)
    REG = DatasetRegistry.load(ROOT, validate_cache_files=False)
    config, FEATURES, CANDIDATES = prepare_features()
    OUT.mkdir(parents=True, exist_ok=True)
    clean = tuple(i for i in REG.encounter_ids if REG.encounters[i]["scan_sum_equal_sensitivity_cohort"])
    assert len(clean) == 121
    official_models = ("prior_constant", "bag_size_logistic", "metadata_logistic", "handcrafted_logistic", "aves_frozen_mil")
    sources = [Path(__file__), ROOT / "configs/modeling_protocol_v1.toml",
        ROOT / "configs/validation_protocol_v1.toml", ROOT / "src/dolphin_behavior_mil/classical.py",
        ROOT / "src/dolphin_behavior_mil/modeling_data.py", ROOT / "src/dolphin_behavior_mil/splitting.py",
        ROOT / "src/dolphin_behavior_mil/statistical_analysis.py",
        ROOT / "results/statistical_analysis/bootstrap_ap_replicates_v1.npz",
        ROOT / "results/statistical_analysis/confirmatory_tests_v1.json"]
    sources += [REG.version_dir / name for name in ("encounters.jsonl", "labels.jsonl", "whistles.jsonl",
                "environment_flat.jsonl", "outer_fold_assignments.jsonl", "inner_fold_assignments.jsonl")]
    sources += [ROOT / f"results/nested_cv/oof/{m}/oof_predictions.jsonl" for m in official_models]
    hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in sources}
    specification = {
        "status": "post_hoc_exploratory", "date": "2026-09-17", "seed": SEED,
        "primary_results_unchanged": True, "endpoint": "pooled encounter-level raw macro-AP",
        "feature_structure": STRUCTURE,
        "feature_quality": [x for x in config["handcrafted_logistic"]["per_whistle_features"] if x not in STRUCTURE],
        "candidate_grid": CANDIDATES, "selection": "mean four-inner-fold AP; within 0.001 tie use smallest candidate ID",
        "training": "original LR, imputation/scaling/vocabulary fitted inside each training partition",
        "new_feature_models": ["structure_logistic", "quality_logistic", "metadata_acoustic_logistic"],
        "count_consistent": {"n": 121, "models": list(official_models[:4]), "refit_and_reselect": True},
        "expanded_campaign_buffers": {"maximum_gap_days": [3, 7], "models": ["prior_constant", "metadata_logistic", "handcrafted_logistic"],
            "rule": "keep original outer test sets; remove training encounters sharing an expanded campaign with test; also buffer inner training against inner validation",
            "dates": "all 258 source encounters; same within-year chaining rule as original"},
        "bootstrap": {"replicates": 10000, "ci": 0.95, "multiplicity_adjusted": False,
            "resampling_unit": "scenario-specific whole campaign blocks",
            "training_uncertainty_included": False, "p_values": "not computed"},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
        "source_sha256": hashes,
    }
    atomic_write_json(OUT / "analysis_specification.json", specification)
    audit = dataset_audit()
    metrics, comparisons, bootstrap_arrays, all_predictions = [], [], {}, {}
    all_encounters = read_rows(REG.version_dir / "encounters.jsonl")
    scenarios = [("primary", REG.encounter_ids, REG.campaign_by_encounter, list(official_models) + specification["new_feature_models"]),
                 ("count_consistent_refit", clean, REG.campaign_by_encounter, list(official_models[:4]))]
    for gap in (3, 7):
        dates = campaign_ids_by_date([r["audio_date"] for r in all_encounters], maximum_gap_days=gap)
        groups = {i: dates[REG.encounters[i]["audio_date"]] for i in REG.encounter_ids}
        scenarios.append((f"gap_{gap}_buffer", REG.encounter_ids, groups,
                          ["prior_constant", "metadata_logistic", "handcrafted_logistic"]))
    for scenario, ids, groups, models in scenarios:
        y = np.stack([REG.labels[i] for i in ids]).astype(int)
        group_order, draws, weights = bootstrap_weights(ids, groups, primary=scenario == "primary")
        atomic_write_json(OUT / f"{scenario}_cohort.json", {"encounter_ids": ids,
            "group_by_encounter": {i: groups[i] for i in ids}, "positive_counts": y.sum(axis=0).tolist()})
        bootstrap_arrays[f"{scenario}__campaign_order"] = np.asarray(group_order)
        bootstrap_arrays[f"{scenario}__campaign_draws"] = draws
        for model_id in models:
            print(f"{scenario}: {model_id}", flush=True)
            p = load_official(model_id) if scenario == "primary" and model_id in official_models else run_nested(scenario, model_id, ids, groups)
            ap = weighted_bootstrap_ap(y, p, weights)
            macro = ap.mean(axis=1)
            point = metric(y, p)
            if scenario == "primary" and model_id in official_models:
                with np.load(ROOT / "results/statistical_analysis/bootstrap_ap_replicates_v1.npz") as z:
                    np.testing.assert_allclose(ap, z[f"raw__{model_id}__per_label_ap"], atol=1e-12, rtol=1e-12)
            lo, hi = interval(macro)
            row = {"scenario": scenario, "model_id": model_id, "n": len(ids), "campaigns": len(group_order),
                   "raw_macro_ap": point, "ci_lower": lo, "ci_upper": hi}
            metrics.append(row)
            bootstrap_arrays[f"{scenario}__{model_id}__macro_ap"] = macro
            all_predictions[(scenario, model_id)] = (point, macro)
            if scenario == "primary" and model_id == "handcrafted_logistic":
                prevalence = (weights @ y) / weights.sum(axis=1)[:, None]
                label_rows = []
                for k, label in enumerate(LABEL_ORDER):
                    aplo, aphi = interval(ap[:, k])
                    dlo, dhi = interval(ap[:, k] - prevalence[:, k])
                    label_rows.append({"label": label, "positive": int(y[:, k].sum()), "n": len(y),
                        "prevalence": float(y[:, k].mean()), "raw_ap": float(average_precision_score(y[:, k], p[:, k])),
                        "ap_ci_lower": aplo, "ap_ci_upper": aphi,
                        "ap_minus_prevalence": float(average_precision_score(y[:, k], p[:, k]) - y[:, k].mean()),
                        "difference_ci_lower": dlo, "difference_ci_upper": dhi,
                        "roc_auc": float(roc_auc_score(y[:, k], p[:, k]))})
                write_csv(OUT / "per_behavior_evidence.csv", label_rows)
            print(f"  raw macro-AP {point:.6f} ({lo:.6f}, {hi:.6f})", flush=True)
        pairs = [("handcrafted_logistic", m) for m in ("prior_constant", "bag_size_logistic", "metadata_logistic", "aves_frozen_mil") if m in models]
        if scenario == "primary":
            pairs += [("metadata_acoustic_logistic", "metadata_logistic"),
                      ("structure_logistic", "prior_constant"), ("quality_logistic", "prior_constant"),
                      ("structure_logistic", "quality_logistic")]
        for a, b in pairs:
            point_a, draws_a = all_predictions[(scenario, a)]
            point_b, draws_b = all_predictions[(scenario, b)]
            lo, hi = interval(draws_a - draws_b)
            comparisons.append({"scenario": scenario, "model_a": a, "model_b": b,
                "delta_raw_macro_ap": point_a - point_b, "ci_lower": lo, "ci_upper": hi})
    write_csv(OUT / "model_metrics.csv", metrics)
    write_csv(OUT / "paired_differences.csv", comparisons)
    np.savez_compressed(OUT / "bootstrap_replicates.npz", **bootstrap_arrays)
    # Validate ties independently even if new LR predictions have none.
    y = np.asarray([[1, 0], [0, 1], [1, 1], [0, 0]])
    weighted_bootstrap_ap(y, np.full_like(y, 0.5, dtype=float),
                          np.asarray([[1, 2, 3, 1], [2, 1, 0, 1]]))
    assert {str(p.relative_to(ROOT)): sha256_file(p) for p in sources} == hashes
    atomic_write_json(OUT / "summary.json", {"status": "complete", "analysis_role": "post_hoc_exploratory",
        "metrics": metrics, "paired_differences": comparisons, "dataset_audit": audit,
        "verification": {"all_declared_scenarios_completed": True, "frozen_sources_unchanged": True,
            "all_inner_and_outer_partitions_encounter_and_campaign_disjoint": True,
            "both_classes_present_in_all_training_and_validation_partitions": True,
            "bootstrap_sklearn_checks_per_model": 20,
            "original_bootstrap_arrays_reproduced_to_1e_minus_12": True,
            "convergence_warnings": 0},
        "artifact_sha256": {str(p.relative_to(OUT)): sha256_file(p) for p in sorted(OUT.rglob("*"))
                            if p.is_file() and p.name != "summary.json"}})
    print(f"Complete: {OUT}", flush=True)


if __name__ == "__main__":
    main()
