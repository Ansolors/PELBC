# Prediction output schema

Original-recording prediction uses `schema_version: pelbc_recording_prediction_v1`.
Pre-extracted clip input uses `pelbc_prediction_v1`.

Recording outputs add `status`, source-recording counts/duration in `input_summary`,
and `extraction` with the manifest path, clip directory, detector hash, confidence,
and padding. Completed predictions retain the behavioral fields below.

If `status` is `no_whistles_detected`, `behavior_scores` and `binary_decisions`
are `null`, `ranked_behaviors` is empty, and quality status is `insufficient_evidence`.
No zero probabilities or negative labels are inferred. Every recording run saves
an extraction manifest and its prediction in a separate run directory.

| Field | Meaning |
|---|---|
| `encounter_id` | User-supplied identifier or input-directory name. |
| `input_summary.whistle_clip_count` | Number of WAV files in the encounter bag. |
| `input_summary.input_set_sha256` | SHA-256 of the ordered relative-path/file-hash manifest. |
| `model.artifact_sha256` | SHA-256 of the exact portable model JSON used. |
| `behavior_scores` | Four independent encounter-level scores in frozen label order. |
| `ranked_behaviors` | The same four scores sorted from highest to lowest. |
| `binary_decisions` | Always `null`; no universal threshold is claimed. |
| `quality_control` | Structured status and non-fatal warnings. |
| `clip_details` | Relative path, hash, audio properties, tonal-detection result, and flags. |
| `interpretation` | Stable guardrails for biological and deployment interpretation. |

## Warning codes

- `sparse_whistle_evidence`: fewer than three clips were supplied.
- `sample_rate_outside_training_regimes`: a compatible sample rate was absent
  from the development cohort.
- `core_tonal_component_not_found`: the deterministic 2--40 kHz audit did not
  find a qualifying component in at least one clip.
- `audio_quality:*`: one of the frozen descriptive quality checks was raised.
- `duplicate_audio_content`: two paths have the same file SHA-256.
- `aggregate_feature_extrapolation`: at least one aggregate lies more than six
  training standard deviations from its mean after imputation.

Warnings do not automatically suppress scores. Applications that require
abstention should define and prospectively validate their own rules.

## CSV output

CSV output contains one row per behavior with encounter ID, processing status, rank, score, clip
count, quality status, warning codes, model ID, and model hash. JSON is preferred
when complete audit information is needed.


For an empty detection result, CSV contains one status row with empty behavior, rank, and score fields.
