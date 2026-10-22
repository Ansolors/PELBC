from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pelbc.cli import _csv_text, main as cli_main
from pelbc.detector import DetectionEvent, DetectorConfig, merge_detections
from pelbc.errors import InputValidationError
from pelbc.predictor import EncounterPredictor
from pelbc.recording import RecordingPredictor
from test_source_only import artificial_model


def write_recording(path: Path, rate: int = 288_000) -> np.ndarray:
    time = np.arange(rate, dtype=np.float64) / rate
    pcm = np.round(6000 * np.sin(2 * np.pi * (8000 * time + 1200 * time ** 2))).astype(np.int16)
    wavfile.write(path, rate, pcm)
    return pcm


class FixedDetector:
    config = DetectorConfig()
    device = "cpu"

    def __init__(self, empty: bool = False, fail: bool = False):
        self.empty = empty
        self.fail = fail

    def detect(self, source: Path, progress=None):
        if self.fail:
            raise RuntimeError("simulated detector failure")
        if progress is not None:
            progress(1, 1, "test")
        if self.empty:
            return []
        return [DetectionEvent(str(source), "W0001", 0.4, 0.6, 8000, 11000, 0.9, [0])]


class RecordingPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.model_directory = Path(temporary.name)
        self.model_path = self.model_directory / "artificial_model.json"
        self.model_path.write_text(json.dumps(artificial_model()), encoding="utf-8")
        # The detector is mocked in these contract tests; this is not a checkpoint.
        self.weights = self.model_directory / "artificial_detector.pt"
        self.weights.write_bytes(b"test placeholder; never loaded by YOLO")

    def recording_predictor(self):
        return RecordingPredictor(
            EncounterPredictor.from_model_path(self.model_path),
            detector_weights=self.weights,
        )

    def test_native_pcm_offsets_and_behavior_scores_match_exported_bag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "encounter"
            source_dir.mkdir()
            (source_dir / "part2").mkdir()
            original = write_recording(source_dir / "source.wav")
            write_recording(source_dir / "part2/source.wav")
            with patch.object(RecordingPredictor, "_detector", return_value=FixedDetector()):
                result = self.recording_predictor().predict(source_dir, recursive=True)
            self.assertEqual(result["input_summary"]["recording_count"], 2)
            self.assertEqual(result["input_summary"]["whistle_clip_count"], 2)
            self.assertEqual(result["status"], "complete")
            manifest = json.loads(Path(result["extraction"]["manifest_path"]).read_text())
            self.assertEqual(len({row["audio"]["clip_file"] for row in manifest["events"]}), 2)
            clips = Path(result["extraction"]["clip_directory"])
            for row in manifest["events"]:
                audio = row["audio"]
                rate, exported = wavfile.read(clips / audio["clip_file"])
                self.assertEqual(rate, 288_000)
                np.testing.assert_array_equal(
                    exported, original[audio["start_sample"]:audio["end_sample_exclusive"]]
                )
            direct = EncounterPredictor.from_model_path(self.model_path).predict_directory(clips)
            self.assertEqual(result["behavior_scores"], direct["behavior_scores"])

    def test_empty_detection_does_not_use_stale_clips_or_assign_zero_probabilities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            write_recording(source)
            output = root / "results"
            stale = output / "old_run/whistles"
            stale.mkdir(parents=True)
            write_recording(stale / "old.wav")
            with patch.object(RecordingPredictor, "_detector", return_value=FixedDetector(empty=True)):
                result = self.recording_predictor().predict(source, work_dir=output)
            self.assertEqual(result["status"], "no_whistles_detected")
            self.assertIsNone(result["behavior_scores"])
            self.assertEqual(result["input_summary"]["whistle_clip_count"], 0)
            rows = list(csv.DictReader(io.StringIO(_csv_text(result))))
            self.assertEqual(rows[0]["score"], "")
            self.assertEqual(rows[0]["status"], "no_whistles_detected")
            self.assertTrue((stale / "old.wav").is_file())

    def test_raw_recording_cli_creates_result_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source.wav", root / "result.json"
            write_recording(source)
            with patch.object(RecordingPredictor, "_detector", return_value=FixedDetector()):
                code = cli_main([
                    "--recording", str(source), "--output", str(output), "--quiet",
                    "--model", str(self.model_path), "--detector-model", str(self.weights),
                ])
            self.assertEqual(code, 0)
            result = json.loads(output.read_text())
            self.assertEqual(len(result["behavior_scores"]), 4)
            self.assertTrue(Path(result["extraction"]["manifest_path"]).is_file())

    def test_failed_detector_has_no_prediction_and_records_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source.wav", root / "results"
            write_recording(source)
            with patch.object(RecordingPredictor, "_detector", return_value=FixedDetector(fail=True)):
                with self.assertRaises(RuntimeError):
                    self.recording_predictor().predict(source, work_dir=output)
            manifests = list(output.glob("run_*/extraction_manifest.json"))
            self.assertEqual(len(manifests), 1)
            self.assertEqual(json.loads(manifests[0].read_text())["status"], "failed")
            self.assertEqual(list(output.glob("run_*/prediction.json")), [])

    def test_incompatible_recording_is_rejected_before_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.wav"
            write_recording(source, rate=48_000)
            with patch.object(RecordingPredictor, "_detector") as detector:
                with self.assertRaises(InputValidationError):
                    self.recording_predictor().predict(source)
                detector.assert_not_called()

    def test_output_cannot_be_mixed_with_source_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_recording(root / "source.wav")
            with self.assertRaises(InputValidationError):
                self.recording_predictor().predict(root, work_dir=root / "output")

    def test_source_only_recording_mode_requires_local_detector_weights(self):
        with self.assertRaisesRegex(InputValidationError, "--detector-model"):
            RecordingPredictor(EncounterPredictor.from_model_path(self.model_path))

    def test_cross_window_association_retains_distinct_events_in_same_window(self):
        rows = [
            DetectionEvent("source.wav", "", 0.4, 0.7, 8000, 11000, 0.8, [0]),
            DetectionEvent("source.wav", "", 0.41, 0.71, 8000, 11000, 0.9, [1]),
            DetectionEvent("source.wav", "", 0.42, 0.69, 8000, 11000, 0.7, [0]),
        ]
        events = merge_detections(rows, DetectorConfig())
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].raw_detection_count, 2)
        self.assertEqual(events[0].contributing_windows, [0, 1])
        self.assertAlmostEqual(events[0].start_s, 0.4)
        self.assertAlmostEqual(events[0].end_s, 0.71)


if __name__ == "__main__":
    unittest.main()
