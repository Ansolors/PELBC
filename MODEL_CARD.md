# PELBC model specification

PELBC predicts encounter-level behavioral context from original Indo-Pacific humpback dolphin recordings. It combines automatic whistle extraction with an acoustic logistic-regression classifier. Trained weights and confidential study data are not distributed in this repository.

## Whistle extraction

The front end uses sliding-window YOLO11s detection, cross-window event association, and native-rate PCM16 export. Defaults are 3-s windows, 1-s hops, confidence 0.62, and 0.25-s padding on each side of a detected event. See [implementation details](docs/DETECTOR_SOURCE.md).

Supply a compatible whistle-trained checkpoint with `--detector-model`. Its SHA-256 and extraction settings are recorded for each run. Detection misses, false detections, and segment boundaries can affect downstream scores. Pipeline tests do not estimate detection performance.

## Acoustic LR

Fourteen deterministic measurements per whistle describe duration, tonal frequency and bandwidth, tonal duration and frame fraction, salience, amplitude, clipping, and low activity. Each is summarized across the encounter by mean, standard deviation, minimum, q25, median, q75, maximum, and missing fraction, giving 112 aggregates. Imputation, standardization, and missing indicators produce 224 inputs for four independent logistic heads.

The portable JSON contract is `pelbc_acoustic_lr_model_v1`. The artifact must supply feature-extraction settings, encoder statistics, model coefficients, and the fixed label order: Feeding, Travelling, Milling, Socializing. Loading validates this contract; the public package supplies no fitted parameters. Model training and export are performed separately from this prediction package.

## Intended use and interpretation

Inputs are one recording or several recordings from the same encounter, or an existing bag of whistle clips. WAV files must be uncompressed mono PCM16 with a sample rate of at least 80 kHz. All prediction processing is local.

The four scores describe encounter-level behavioral context. They need not sum to one and do not identify individual callers, individual-whistle behavior, or future behavior. No prospective probability calibration or universal binary threshold is claimed. Independent validation is needed for new survey conditions, populations, and species.

No detected whistles returns `behavior_scores: null` with status `no_whistles_detected`. Sparse bags, duplicates, unfamiliar sampling rates, acoustic quality issues, and feature extrapolation generate diagnostics. They do not automatically suppress scores.

The paper evaluates the behavioral modeling procedure using campaign-grouped nested validation. It reports Acoustic LR as the best of 11 primary methods, with raw macro-AP 0.710, raw macro ROC AUC 0.815, and calibrated macro Brier score 0.145 across four behaviors, 234 encounters, and 4,126 clips. The final full-data refit is an inference artifact, not a separately validated deployment model. Public artificial-input checks validate software behavior only; the repository alone cannot reproduce the study's numerical results.
