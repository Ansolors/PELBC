#!/usr/bin/env python3
"""Generate non-biological chirps for a CLI installation smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import chirp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("synthetic_encounter"),
    )
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_rate_hz = 96000
    for index in range(3):
        duration = 0.18 + 0.02 * index
        time = np.arange(int(round(duration * sample_rate_hz)), dtype=np.float64)
        time /= sample_rate_hz
        signal = chirp(
            time,
            f0=7000.0 + 350.0 * index,
            f1=12000.0 + 500.0 * index,
            t1=duration,
            method="quadratic",
        )
        envelope = np.sin(np.pi * np.arange(len(time)) / max(len(time) - 1, 1)) ** 2
        waveform = np.clip(0.25 * envelope * signal, -1.0, 1.0)
        path = output_dir / f"synthetic_whistle_{index + 1:02d}.wav"
        wavfile.write(path, sample_rate_hz, np.round(waveform * 32767.0).astype(np.int16))
    print(output_dir)
    print("These chirps are only an installation test and have no biological interpretation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
