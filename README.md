# PELBC

**PELBC** (Predicting Encounter-Level Behavioral Context) is software for predicting Indo-Pacific humpback dolphin behavioral context from encounter recordings. It automatically detects and extracts whistle segments, assembles an encounter bag, and outputs **Feeding**, **Travelling**, **Milling**, and **Socializing** scores using Acoustic LR. All processing runs locally.

[中文说明](README_zh-CN.md)

```text
Encounter recordings → automatic whistle detection → cross-window event association
→ native-rate whistle clips → encounter-level acoustic summaries → four behavioral scores
```

This repository maintains one PELBC implementation. Recording input and pre-extracted clip input are two entry modes of the same software.

## Contents

- `src/pelbc/`: whistle detection and extraction, acoustic measurements, encounter aggregation, portable logistic inference, and command-line interface.
- `examples/` and `tests/`: synthetic-input generation and software checks.
- `docs/` and `MODEL_CARD.md`: extraction settings, input requirements, model contract, and output format.

This is a **source-only distribution**. Confidential recordings, annotations, encounter metadata, derived study data, prediction results, and trained model parameters are not included. Supply a compatible local Acoustic LR JSON model via `--model` and, for recording input, a locally trained whistle detector checkpoint via `--detector-model`. A generic object-detection checkpoint is not a substitute for the whistle detector. PELBC does not download either model automatically.

This repository distributes the prediction software. The paper, experiment and training scripts, study results, and private model-export workflow are maintained separately and are not part of this distribution.

## Install

Python 3.10–3.13 is supported. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

Detector inference uses Ultralytics 8.4.102. Automatic device selection prefers MPS, then CUDA, then CPU; use `--device cpu` to select CPU explicitly.

## Predict from encounter recordings

```bash
pelbc-predict \
  --recording /path/to/encounter_001.wav \
  --model /path/to/private/acoustic_model.json \
  --detector-model /path/to/private/whistle_detector.pt \
  --work-dir /path/to/local/encounter_001_results \
  --output /path/to/local/prediction.json
```

For several recordings from **the same encounter**, replace `--recording` with `--recordings-dir /path/to/encounter_001_recordings`. Use `--encounter-id encounter_001` to set an identifier. Add `--recursive` only when every nested WAV belongs to that encounter. Do not mix encounters in one invocation.

Inputs must be uncompressed mono PCM16 WAV at a sample rate of at least 80 kHz. Detection uses 3-s windows, a 1-s hop, confidence 0.62, and cross-window event association. Each event is exported with 0.25 s of context on either side, preserving source PCM samples and native sample rate. Detection images cover 0–30 kHz; downstream tonal measurements cover 2–40 kHz. Details are in [whistle extraction](docs/DETECTOR_SOURCE.md).

Each invocation creates a fresh directory:

```text
encounter_001_results/run_.../
├── whistles/                   Native-rate whistle clips
├── extraction_manifest.json    Sources, sample offsets, detections, and model hash
└── prediction.json             Encounter-level prediction
```

If `--work-dir` is omitted, the output directory is `<input-name>_pelbc` beside the input. It must be outside a source-recording directory. Separate run directories prevent old clips from entering the current bag.

Use `--output prediction.csv` for a compact table; without `--output`, JSON is also printed to standard output. `--quiet` suppresses progress. If no whistles are detected, the result has status `no_whistles_detected` and `behavior_scores: null`; it does not assign four negative labels. See the [output schema](docs/OUTPUT_SCHEMA.md).

## Existing whistle clips

```bash
pelbc-predict \
  --input-dir /path/to/encounter_001_whistles \
  --model /path/to/private/acoustic_model.json \
  --output /path/to/local/prediction.json
```

This mode skips detection and scores the supplied encounter bag. It uses the same feature extraction and behavioral model, and does not require a detector checkpoint.

## Python API

```python
from pelbc import EncounterPredictor, RecordingPredictor

behavior = EncounterPredictor.from_model_path("/path/to/private/acoustic_model.json")
software = RecordingPredictor(
    behavior,
    detector_weights="/path/to/private/whistle_detector.pt",
)
result = software.predict("/path/to/encounter_001.wav")
print(result["behavior_scores"])

# Alternatively, score an existing whistle bag from one encounter.
result = behavior.predict_directory("/path/to/encounter_001_whistles")
```

## Interpretation

The accompanying manuscript reports **Acoustic LR** as the best of 11 primary methods, with raw macro-AP **0.710**, raw macro ROC AUC 0.815, and calibrated macro Brier score 0.145. The study covers **four behavioral categories, 234 encounters, and 4,126 whistle clips**. These are manuscript evaluation summaries, not a separate validation of a locally supplied model artifact.

The prediction unit is the encounter. Four independent multilabel scores may co-occur and need not sum to one. They do not identify callers or assign behavior to individual whistles. They are not prospectively calibrated probabilities for a new survey, and no universal binary threshold is supplied. See the [model specification](MODEL_CARD.md).

## Check the software

```bash
python -m unittest discover -s tests -v
```

Tests use artificial audio and model parameters, with a simulated detector for pipeline checks. They cover native PCM preservation, event association, encounter assembly, recording and clip inputs, missing models, empty detections, failed runs, and JSON/CSV output. They do not measure biological prediction performance or detector accuracy. Citation metadata is in [CITATION.cff](CITATION.cff).
