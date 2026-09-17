# Core research code

This directory contains the paper's analysis implementations and method configurations. It starts from privately prepared encounter tables and whistle clips; raw-data curation and confidential study records are not distributed. The scripts retain the study's cohort and schema checks, including the primary 234-encounter, 4,126-whistle cohort. Applying them to a different dataset requires adapting those contracts.

## Setup

From the repository root, install the research package in editable mode so that scripts, configurations, and source remain together:

```bash
python -m pip install -e "./research[train]"
cd research
cp configs/paths.example.toml configs/paths.toml
```

Edit the ignored `configs/paths.toml` locally. Raw-data paths must point outside the repository. Prepared tables and generated outputs live in ignored `data/` and `results/` directories. The runner expects `data/processed/v0.4.0/` with `encounters.jsonl`, `labels.jsonl`, `whistles.jsonl`, `environment_flat.jsonl`, `outer_fold_assignments.jsonl`, and `inner_fold_assignments.jsonl`. Required fields and validation are defined in `src/dolphin_behavior_mil/modeling_data.py`; whistle records also reference local feature caches. Additional analyses require their corresponding private audit tables, prediction files, or checkpoints.

No data records, feature caches, split assignments, or checkpoints are supplied. The repository alone cannot reproduce the reported numerical results.

## Analysis entry points

| Purpose | Implementation |
| --- | --- |
| Audio audit and log-mel features | `audio_audit.py`, `audio_preprocessing.py`; `scripts/cache_logmel_features.py` |
| Acoustic and control baselines | `classical.py` |
| Convolutional and frozen-embedding multiple-instance models | `mil_models.py`, `pretrained_embeddings.py` |
| Grouped splits, fold-local preparation, and training | `splitting.py`, `modeling_data.py`, `training.py`, `nested_validation.py` |
| Nested validation | `scripts/run_nested_experiments.py` |
| Primary statistics | `scripts/run_primary_statistics.py` |
| Cohort, bag, bandwidth, prefix, and stress analyses | `scripts/run_*sensitivities.py`, `scripts/run_bandwidth_sensitivity.py`, `scripts/run_prefix_evaluation.py`, `scripts/run_stress_experiments.py` |
| Additional diagnostics | `scripts/run_posthoc_revision_diagnostics.py`, `scripts/summarize_handcrafted_coefficients.py` |

Module names in the table refer to `src/dolphin_behavior_mil/`. Configurations in `configs/` specify the analysis settings; the CNN execution settings are in `cnn_compute_protocol_v1.toml`.

To inspect the runner or construct the model registry without study data:

```bash
python scripts/build_model_registry_v1.py
python scripts/run_nested_experiments.py --help
```

Run feature caching before training, and primary statistics after the nested-validation outputs are complete. Sensitivity and diagnostic scripts consume those local outputs. Public third-party encoder sources and asset references are recorded in `configs/pretrained_encoders_v1.toml`; their weights are not bundled and must be obtained separately under the providers' terms.

The numeric-label cohort sensitivity additionally requires an ignored `configs/cohort_bag_private.toml` containing `additional_encounters`, `additional_encounter_campaign`, `additional_encounter_outer_fold`, and `expected_extra_instance_count`. These study-specific identifiers and counts must be supplied locally.

## Synthetic checks

```bash
python -m unittest discover -s tests -v
```

Tests use artificial inputs only. They check model pooling and evaluation behavior, not reproduction of study results. Do not commit private configuration files, data, derived outputs, or trained models.
