#!/usr/bin/env python3
"""Build the frozen 192-kHz/0.5-90-kHz sensitivity feature cache."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.io import wavfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dolphin_behavior_mil.audio_preprocessing import (  # noqa: E402
    extract_frozen_logmel,
    feature_sha256,
)
from dolphin_behavior_mil.experiment import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    load_toml,
    sha256_file,
)
from dolphin_behavior_mil.paths import load_paths  # noqa: E402


SOURCE_PATH = PROJECT_ROOT / "data/processed/v0.4.0/whistles.jsonl"
PREPROCESS_PATH = PROJECT_ROOT / "configs/audio_preprocessing_v1.toml"
EXECUTION_PATH = PROJECT_ROOT / "configs/bandwidth_sensitivity_execution_v1.toml"
CACHE_ROOT = PROJECT_ROOT / "results/cache/logmel_192k_90k_v1"
OUTPUT_ROOT = PROJECT_ROOT / "results/sensitivity/bandwidth"
MANIFEST_PATH = OUTPUT_ROOT / "logmel_192k_manifest_v1.jsonl"
SUMMARY_PATH = OUTPUT_ROOT / "logmel_192k_summary_v1.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def worker(
    payload: tuple[dict[str, Any], str, dict[str, Any], bool]
) -> dict[str, Any]:
    row, raw_root, preprocessing, force = payload
    output_path = (
        CACHE_ROOT / str(row["encounter_id"]) / f"{row['whistle_id']}.npy"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file() and not force:
        feature = np.load(output_path, allow_pickle=False)
    else:
        sample_rate, pcm = wavfile.read(
            Path(raw_root) / str(row["clip_filename"]), mmap=True
        )
        _, extracted = extract_frozen_logmel(
            np.asarray(pcm),
            int(sample_rate),
            preprocessing,
            bandwidth_sensitivity=True,
        )
        feature = np.asarray(extracted, dtype=np.float16, order="C")
        temporary = output_path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.save(handle, feature, allow_pickle=False)
        temporary.replace(output_path)
    if (
        feature.dtype != np.float16
        or feature.ndim != 2
        or feature.shape[0] != 192
        or not np.all(np.isfinite(feature))
        or float(np.min(feature)) < -80.00001
        or float(np.max(feature)) > 0.00001
    ):
        raise ValueError(f"invalid 192-kHz feature: {row['whistle_id']}")
    return {
        "preprocessing_id": "whistle_logmel_192k_90k_v1",
        "whistle_id": str(row["whistle_id"]),
        "encounter_id": str(row["encounter_id"]),
        "clip_sha256": str(row["clip_sha256"]),
        "native_sample_rate_hz": int(row["sample_rate_hz"]),
        "feature_shape": list(feature.shape),
        "feature_dtype": str(feature.dtype),
        "feature_min": float(np.min(feature)),
        "feature_max": float(np.max(feature)),
        "feature_content_sha256": feature_sha256(feature),
        "cache_path_relative_to_project": str(output_path.relative_to(PROJECT_ROOT)),
        "cache_file_bytes": output_path.stat().st_size,
        "cache_file_sha256": sha256_file(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    execution = load_toml(EXECUTION_PATH)
    preprocessing = load_toml(PREPROCESS_PATH)
    bandwidth = preprocessing["bandwidth_sensitivity"]
    if (
        int(bandwidth["sample_rate_hz"]) != int(execution["sample_rate_hz"])
        or int(bandwidth["n_mels"]) != int(execution["n_mels"])
        or [float(bandwidth["f_min_hz"]), float(bandwidth["f_max_hz"])]
        != [float(value) for value in execution["frequency_range_hz"]]
    ):
        raise RuntimeError("bandwidth execution and preprocessing configs disagree")
    rows = [
        row
        for row in read_jsonl(SOURCE_PATH)
        if bool(row.get("primary_analysis_cohort"))
        and bool(row.get("main_discrete_whistle_instance"))
    ]
    rows.sort(key=lambda row: str(row["whistle_id"]))
    if len(rows) != int(execution["main_instance_count"]):
        raise RuntimeError("expected exactly 4,126 main instances")
    raw_root = str(load_paths().whistle_clips)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    payloads = [
        (row, raw_root, preprocessing, bool(args.force)) for row in rows
    ]
    if args.workers == 1:
        iterator = map(worker, payloads)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=int(args.workers))
        iterator = executor.map(worker, payloads, chunksize=1)
    outputs: list[dict[str, Any]] = []
    try:
        for index, output in enumerate(iterator, start=1):
            outputs.append(output)
            if index % 100 == 0 or index == len(rows):
                print(f"cached 192-kHz features {index}/{len(rows)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(MANIFEST_PATH, outputs)
    summary = {
        "status": "complete",
        "preprocessing_id": "whistle_logmel_192k_90k_v1",
        "feature_count": len(outputs),
        "encounter_count": len({row["encounter_id"] for row in outputs}),
        "sample_rate_hz": int(bandwidth["sample_rate_hz"]),
        "frequency_range_hz": [
            float(bandwidth["f_min_hz"]),
            float(bandwidth["f_max_hz"]),
        ],
        "n_mels": int(bandwidth["n_mels"]),
        "all_time_frames": int(sum(row["feature_shape"][1] for row in outputs)),
        "cache_total_bytes": int(sum(row["cache_file_bytes"] for row in outputs)),
        "all_features_float16": all(row["feature_dtype"] == "float16" for row in outputs),
        "all_features_192_mels": all(row["feature_shape"][0] == 192 for row in outputs),
        "all_features_bounded_minus80_to_zero": all(
            row["feature_min"] >= -80.0 and row["feature_max"] <= 0.0
            for row in outputs
        ),
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in (SOURCE_PATH, PREPROCESS_PATH, EXECUTION_PATH, Path(__file__))
        },
        "manifest_sha256": sha256_file(MANIFEST_PATH),
    }
    atomic_write_json(SUMMARY_PATH, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
