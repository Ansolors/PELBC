#!/usr/bin/env python3
"""Build the complete rebuildable 96 kHz log-mel cache and content manifest."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from typing import Any

import numpy as np
from scipy.io import wavfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dolphin_behavior_mil.audio_preprocessing import (  # noqa: E402
    extract_frozen_logmel,
    feature_sha256,
)
from dolphin_behavior_mil.paths import load_paths  # noqa: E402


SOURCE_PATH = PROJECT_ROOT / "data/processed/v0.2.0/whistles.jsonl"
CONFIG_PATH = PROJECT_ROOT / "configs/audio_preprocessing_v1.toml"
OUTPUT_DIR = PROJECT_ROOT / "results/audio_audit"
MANIFEST_PATH = OUTPUT_DIR / "logmel_cache_manifest_v1.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "logmel_cache_summary_v1.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _worker(
    payload: tuple[dict[str, Any], str, str, dict[str, Any], bool]
) -> dict[str, Any]:
    row, raw_root, cache_root, config, force = payload
    output_path = (
        Path(cache_root) / str(row["encounter_id"]) / f"{row['whistle_id']}.npy"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        feature = np.load(output_path, allow_pickle=False)
    else:
        sample_rate, pcm = wavfile.read(
            Path(raw_root) / str(row["clip_filename"]),
            mmap=True,
        )
        _, extracted = extract_frozen_logmel(
            np.asarray(pcm),
            int(sample_rate),
            config,
        )
        feature = np.asarray(extracted, dtype=np.float16, order="C")
        temporary_path = output_path.with_suffix(".tmp")
        with temporary_path.open("wb") as handle:
            np.save(handle, feature, allow_pickle=False)
        temporary_path.replace(output_path)
    if feature.dtype != np.float16 or feature.ndim != 2 or feature.shape[0] != 128:
        raise ValueError(f"invalid cached feature shape/dtype for {row['whistle_id']}")
    relative = output_path.relative_to(PROJECT_ROOT)
    return {
        "preprocessing_id": config["preprocessing"]["preprocessing_id"],
        "whistle_id": row["whistle_id"],
        "encounter_id": row["encounter_id"],
        "clip_sha256": row["clip_sha256"],
        "primary_analysis_cohort": bool(row["primary_analysis_cohort"]),
        "loop_type": row["loop_type"],
        "feature_shape": list(feature.shape),
        "feature_dtype": str(feature.dtype),
        "feature_min": float(np.min(feature)),
        "feature_max": float(np.max(feature)),
        "feature_content_sha256": feature_sha256(feature),
        "cache_path_relative_to_project": str(relative),
        "cache_file_bytes": output_path.stat().st_size,
        "cache_file_sha256": sha256(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    rows = read_jsonl(SOURCE_PATH)
    if len(rows) != 4695:
        raise RuntimeError("expected 4,695 whistle source rows")
    with CONFIG_PATH.open("rb") as handle:
        config = tomllib.load(handle)
    paths = load_paths()
    cache_root = PROJECT_ROOT / config["storage"]["cache_root"]
    cache_root.mkdir(parents=True, exist_ok=True)
    payloads = [
        (row, str(paths.whistle_clips), str(cache_root), config, args.force)
        for row in rows
    ]
    if args.workers == 1:
        iterator = map(_worker, payloads)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        iterator = executor.map(_worker, payloads, chunksize=1)
    outputs = []
    try:
        for index, output in enumerate(iterator, start=1):
            outputs.append(output)
            if index % 100 == 0 or index == len(rows):
                print(f"cached {index}/{len(rows)} features", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        for row in outputs:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )
    primary = [row for row in outputs if row["primary_analysis_cohort"]]
    main = [row for row in primary if row["loop_type"] is None]
    summary = {
        "preprocessing_id": config["preprocessing"]["preprocessing_id"],
        "source_dataset_version": config["preprocessing"]["source_dataset_version"],
        "all_feature_count": len(outputs),
        "primary_feature_count": len(primary),
        "main_discrete_feature_count": len(main),
        "all_time_frames": int(sum(row["feature_shape"][1] for row in outputs)),
        "primary_time_frames": int(sum(row["feature_shape"][1] for row in primary)),
        "main_discrete_time_frames": int(
            sum(row["feature_shape"][1] for row in main)
        ),
        "cache_total_bytes": int(sum(row["cache_file_bytes"] for row in outputs)),
        "feature_shape_time_quantiles": {
            key: float(value)
            for key, value in zip(
                ("q00", "q01", "q05", "q50", "q95", "q99", "q100"),
                np.quantile(
                    [row["feature_shape"][1] for row in outputs],
                    (0.0, 0.01, 0.05, 0.50, 0.95, 0.99, 1.0),
                ),
                strict=True,
            )
        },
        "all_features_float16": all(row["feature_dtype"] == "float16" for row in outputs),
        "all_features_128_mels": all(row["feature_shape"][0] == 128 for row in outputs),
        "all_features_bounded_minus80_to_zero": all(
            row["feature_min"] >= -80.0 and row["feature_max"] <= 0.0
            for row in outputs
        ),
        "input_sha256": {
            "whistles_jsonl": sha256(SOURCE_PATH),
            "audio_preprocessing_config": sha256(CONFIG_PATH),
        },
        "manifest_sha256": sha256(MANIFEST_PATH),
        "cache_policy": config["storage"],
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
