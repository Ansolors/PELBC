# PELBC

Predicting Encounter-Level Behavioral Context (PELBC): source code for encounter-level multilabel analysis of dolphin whistles and the accompanying offline inference package.

## Contents

- `src/pelbc/`: acoustic feature extraction, encounter aggregation, portable logistic inference, and command-line interface.
- `research/`: the paper's core preprocessing, models, grouped validation, statistical analysis, and sensitivity analyses. See the [research instructions](research/README.md).
- `tools/export_acoustic_lr.py`: export a portable model from a private, locally prepared research project.
- `examples/` and `tests/`: synthetic-input generation and software checks.

This is a source-only release. Confidential recordings, annotations, metadata, derived features, predictions, analysis results, and trained model parameters are **not included**. Reproducing the paper's results requires the private research data. Running inference requires a compatible model supplied locally; the package does not download one.

## Install and use

Python 3.10–3.13 is supported. From the repository root:

```bash
python -m pip install .
pelbc-predict --model /path/to/private/model.json \
  --input-dir /path/to/one/encounter \
  --output /path/to/local/prediction.json
```

The input directory must contain pre-extracted mono PCM16 WAV whistle clips from one encounter. JSON and CSV outputs are supported. The four independent scores correspond to Feeding, Travelling, Milling, and Socializing. They describe encounter context, not individual whistles or callers, and are not prospectively calibrated probabilities. This package does not detect whistles in continuous recordings.

The Python API uses the same local model:

```python
from pelbc import EncounterPredictor

predictor = EncounterPredictor.from_model_path("/path/to/private/model.json")
result = predictor.predict_directory("/path/to/one/encounter")
```

For a locally prepared research project, model export additionally requires the research dependencies, its private tables, and completed model-selection and statistical-analysis outputs:

```bash
python -m pip install ".[export]"
python tools/export_acoustic_lr.py --research-project /path/to/private/research
```

The default export directory, `private_models/`, is excluded from Git. Keep all inputs, models, and generated outputs private.

## Check the software

```bash
python -m unittest discover -s tests -v
```

These checks create artificial audio and model parameters in temporary directories; they do not use or validate the confidential study data. Citation metadata is in `CITATION.cff`.
