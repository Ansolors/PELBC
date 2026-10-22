# Whistle extraction implementation

[`pelbc/detector.py`](../src/pelbc/detector.py) detects whistle events in original recordings. [`pelbc/recording.py`](../src/pelbc/recording.py) exports native-rate clips, assembles the encounter bag, and runs behavioral prediction.

The detector processes 3-s windows with 1-s hops. Each window is resampled to 96 kHz for a Hann-window spectrogram using a 20-ms window and a 2.5-ms hop. The 0–30 kHz region is rendered as a 512 × 256 image and supplied to the whistle-trained YOLO11s detector. Defaults are confidence 0.62 and NMS IoU 0.70.

Window-relative detections are mapped to source-recording time. Cross-window association requires frequency overlap and sufficient temporal IoU, overlap, or proximity. Detections from the same analysis window remain separate; merged events retain their contributing windows. The event time span covers the associated detections, and frequency bounds use confidence weighting. Exact settings are defined by `DetectorConfig`.

Each event is exported with 0.25 s of context on either side. Start and end times become native sample indices using floor and ceiling, clipped to recording boundaries. Export preserves the source PCM samples and sample rate. Resampling is used only for detector images, not for exported clips.

Each run records source files and hashes, sample offsets, event confidence, detector configuration, and checkpoint hash in `extraction_manifest.json`. Separate source recordings are detected independently, and their segments are combined only within the supplied encounter. A fresh run directory prevents stale clips from entering the bag.

The public repository contains extraction code but no trained checkpoint. Pass a compatible local checkpoint through `--detector-model` or `RecordingPredictor(..., detector_weights=...)`. No detected events produces an explicit no-evidence result; a failed run records its failure without generating a behavioral prediction.
