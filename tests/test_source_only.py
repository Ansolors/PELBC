"""Source-only package checks with artificial audio and model parameters."""
from __future__ import annotations

import contextlib
import csv
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile
from scipy.signal import chirp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pelbc import EncounterPredictor
from pelbc.cli import main as cli_main
from pelbc.errors import InputValidationError, ModelValidationError
from pelbc.features import AGGREGATION_STATISTICS, PER_WHISTLE_FEATURES, aggregate_whistles
from pelbc.model import MODEL_SCHEMA, PortableAcousticLR

# Public algorithm settings; no fitted state is used here.
SETTINGS = {'audit': {'audit_id': 'native_audio_quality_spectral_v1',
           'source_dataset_version': '0.2.0',
           'window_seconds': 0.010666666666666666,
           'hop_fraction': 0.5,
           'max_frames_per_clip': 512,
           'amplitude_percentile_sample_cap': 200000,
           'spectral_baseline_quantile': 0.2,
           'analysis_lower_hz': 500.0,
           'core_tonal_lower_hz': 2000.0,
           'core_tonal_upper_hz': 40000.0,
           'ultrasonic_tonal_lower_hz': 40000.0,
           'ultrasonic_tonal_upper_hz': 96000.0,
           'supra96_tonal_lower_hz': 96000.0,
           'supra96_tonal_upper_hz': 136000.0,
           'tonal_smoothing_time_sigma_frames': 0.8,
           'tonal_smoothing_frequency_sigma_bins': 1.0,
           'tonal_local_baseline_width_hz': 3000.0,
           'tonal_salience_threshold_db': 3.0,
           'tonal_minimum_duration_seconds': 0.025,
           'retention_cutoffs_hz': [8000,
                                    16000,
                                    24000,
                                    32000,
                                    40000,
                                    48000,
                                    64000,
                                    80000,
                                    96000],
           'frequency_quantiles': [0.5, 0.9, 0.95, 0.99]},
 'descriptive_flags': {'exact_clipping_min_fraction': 0.0,
                       'near_clipping_min_fraction': 0.001,
                       'near_clipping_amplitude_fraction': 0.99,
                       'large_dc_offset_fraction_full_scale': 0.01,
                       'very_low_rms_dbfs': -60.0,
                       'large_zero_fraction': 0.5,
                       'low_tonal_component_frame_fraction': 0.01,
                       'nonloop_long_duration_seconds': 3.0}}


def artificial_model():
    names = list(aggregate_whistles([{key: 0.0 for key in PER_WHISTLE_FEATURES}]))
    output_names = [value for name in names for value in (name, name + "__missing")]
    return {
        "schema_version": MODEL_SCHEMA,
        "model_id": "artificial_test_only",
        "model_type": "synthetic_logistic_heads",
        "label_order": ["Feeding", "Travelling", "Milling", "Socializing"],
        "feature_extraction": {
            "per_whistle_features": list(PER_WHISTLE_FEATURES),
            "aggregation_statistics": list(AGGREGATION_STATISTICS),
            "add_missing_fraction_per_feature": True,
            **SETTINGS,
        },
        "encoder": {
            "numeric_features": names,
            "numeric_medians": [0.0] * len(names),
            "numeric_means": [0.0] * len(names),
            "numeric_standard_deviations": [1.0] * len(names),
            "output_feature_names": output_names,
        },
        "classifier": {"heads": [
            {"kind": "logistic_regression", "coefficient": [[0.0] * len(output_names)],
             "intercept": [index * 0.2]}
            for index in range(4)
        ]},
        "training": {"native_sample_rates_hz": [96000], "encounter_count": 0},
        "score_semantics": {"type": "artificial_test_only"},
    }


class SourceOnlyPackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.audio = self.root / "artificial_encounter"
        self.audio.mkdir()
        rate = 96000
        time = np.arange(int(rate * 0.2)) / rate
        signal = chirp(time, f0=7000, f1=12000, t1=0.2)
        waveform = 0.25 * np.sin(np.pi * np.arange(len(time)) / (len(time) - 1)) ** 2 * signal
        self.pcm = np.round(waveform * 32767).astype(np.int16)
        wavfile.write(self.audio / "artificial.wav", rate, self.pcm)
        self.model_path = self.root / "artificial_model.json"
        self.model_path.write_text(json.dumps(artificial_model()), encoding="utf-8")

    def test_default_model_is_not_bundled(self):
        with self.assertRaisesRegex(ModelValidationError, "no trained model"):
            EncounterPredictor()

    def test_inference_is_deterministic_with_expected_artificial_scores(self):
        predictor = EncounterPredictor.from_model_path(self.model_path)
        first = predictor.predict_directory(self.audio)
        second = predictor.predict_directory(self.audio)
        self.assertEqual(first, second)
        expected = [1.0 / (1.0 + math.exp(-index * 0.2)) for index in range(4)]
        np.testing.assert_allclose(list(first["behavior_scores"].values()), expected)
        self.assertIsNone(first["binary_decisions"])
        self.assertEqual(first["input_summary"]["whistle_clip_count"], 1)

    def test_nonfinite_model_parameters_are_rejected(self):
        record = artificial_model()
        record["classifier"]["heads"][0]["intercept"] = [float("nan")]
        self.model_path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaises(ModelValidationError):
            PortableAcousticLR.load(self.model_path)

    def test_cli_writes_json_and_csv(self):
        for suffix in ("json", "csv"):
            output = self.root / ("prediction." + suffix)
            status = cli_main(["--model", str(self.model_path), "--input-dir", str(self.audio),
                               "--output", str(output), "--quiet"])
            self.assertEqual(status, 0)
            if suffix == "json":
                self.assertEqual(len(json.loads(output.read_text())["behavior_scores"]), 4)
            else:
                with output.open(newline="") as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), 4)

    def test_cli_requires_explicit_local_model(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                cli_main(["--input-dir", str(self.audio)])
        self.assertEqual(caught.exception.code, 2)

    def test_stereo_is_rejected(self):
        wavfile.write(self.audio / "stereo.wav", 96000, np.column_stack((self.pcm, self.pcm)))
        with self.assertRaises(InputValidationError):
            EncounterPredictor.from_model_path(self.model_path).predict_directory(self.audio)


if __name__ == "__main__":
    unittest.main()
