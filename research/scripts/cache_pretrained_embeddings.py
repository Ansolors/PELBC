#!/usr/bin/env python3
"""Generate complete, auditable PANNs and AVES-bio frozen embeddings."""

from __future__ import annotations

import argparse
import gc
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import scipy
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dolphin_behavior_mil.modeling_data import (  # noqa: E402
    DatasetRegistry,
    FrozenEmbeddingRegistry,
)
from dolphin_behavior_mil.pretrained_embeddings import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    build_extractor,
    cache_encoder_embeddings,
    load_pretrained_config,
    main_whistle_rows,
    sha256_file,
    validate_encoder_assets,
)


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def runtime_environment(device: str, threads: int) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": device,
        "torch_num_threads": threads,
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "torchaudio": package_version("torchaudio"),
            "torchlibrosa": package_version("torchlibrosa"),
            "panns-inference": package_version("panns-inference"),
            "esp-aves": package_version("esp-aves"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=("panns", "aves", "all"), default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.torch_threads < 1:
        raise SystemExit("--torch-threads must be positive")
    if args.device != "cpu":
        raise SystemExit("canonical caches are frozen to --device cpu")

    torch.set_num_threads(args.torch_threads)
    torch.use_deterministic_algorithms(True)
    config = load_pretrained_config(PROJECT_ROOT)
    dataset = DatasetRegistry.load(PROJECT_ROOT)
    whistle_rows = main_whistle_rows(dataset)
    encoder_keys = ("panns", "aves") if args.encoder == "all" else (args.encoder,)
    output_root = PROJECT_ROOT / str(config["storage"]["manifest_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    environment = runtime_environment(args.device, args.torch_threads)
    environment["protocol_config_sha256"] = sha256_file(
        PROJECT_ROOT / "configs/pretrained_encoders_v1.toml"
    )
    atomic_write_json(
        PROJECT_ROOT / str(config["storage"]["runtime_environment"]), environment
    )

    final_summaries = {}
    for encoder_key in encoder_keys:
        encoder_config = config["encoders"][encoder_key]
        assets = validate_encoder_assets(PROJECT_ROOT, encoder_config)
        print(f"[{encoder_key}] assets verified; loading model", flush=True)
        extractor = build_extractor(
            PROJECT_ROOT, encoder_config, device=args.device
        )

        def progress(index: int, total: int, row: dict[str, Any]) -> None:
            if index % 50 == 0 or index == total:
                print(
                    f"[{encoder_key}] cached/verified {index}/{total}: {row['whistle_id']}",
                    flush=True,
                )

        manifest_rows, summary = cache_encoder_embeddings(
            project_root=PROJECT_ROOT,
            protocol_config=config,
            encoder_config=encoder_config,
            whistle_rows=whistle_rows,
            extractor=extractor,
            force=args.force,
            progress=progress,
        )
        manifest_path = PROJECT_ROOT / str(encoder_config["manifest_path"])
        summary_path = PROJECT_ROOT / str(encoder_config["summary_path"])
        atomic_write_jsonl(manifest_path, manifest_rows)
        summary["manifest_path_relative_to_project"] = str(
            manifest_path.relative_to(PROJECT_ROOT)
        )
        summary["manifest_sha256"] = sha256_file(manifest_path)
        summary["assets"] = assets
        summary["runtime_environment_path_relative_to_project"] = str(
            (PROJECT_ROOT / str(config["storage"]["runtime_environment"])).relative_to(
                PROJECT_ROOT
            )
        )
        atomic_write_json(summary_path, summary)
        registry = FrozenEmbeddingRegistry.load(
            PROJECT_ROOT, manifest_path, dataset, verify_files=True
        )
        if registry.embedding_dimension != int(encoder_config["embedding_dimension"]):
            raise RuntimeError("post-write embedding registry dimension mismatch")
        final_summaries[encoder_key] = summary
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        del registry, extractor
        gc.collect()

    print(
        json.dumps(
            {"status": "PASS", "encoders": final_summaries},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
