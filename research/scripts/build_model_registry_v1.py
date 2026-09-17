#!/usr/bin/env python3
"""Freeze the Step-8 implementation contract and required-model registry."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dolphin_behavior_mil.experiment import (  # noqa: E402
    atomic_write_json,
    build_model_registry,
    load_toml,
    runtime_environment,
    sha256_file,
    validate_modeling_protocol,
)


def main() -> None:
    configs = {
        "modeling": PROJECT_ROOT / "configs" / "modeling_protocol_v1.toml",
        "method_scope": PROJECT_ROOT / "configs" / "method_scope_v1.toml",
        "validation": PROJECT_ROOT / "configs" / "validation_protocol_v1.toml",
        "audio": PROJECT_ROOT / "configs" / "audio_preprocessing_v1.toml",
    }
    parsed = {key: load_toml(path) for key, path in configs.items()}
    contract = validate_modeling_protocol(
        parsed["modeling"],
        parsed["method_scope"],
        parsed["validation"],
        parsed["audio"],
    )
    registry = build_model_registry(parsed["modeling"])
    output_dir = PROJECT_ROOT / "results" / "model_implementation"
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output_dir / "model_registry_v1.json",
        {
            "protocol_id": parsed["modeling"]["protocol"]["protocol_id"],
            "required_model_count": len(registry),
            "models": registry,
        },
    )
    atomic_write_json(output_dir / "modeling_contract_checks_v1.json", contract)
    atomic_write_json(output_dir / "runtime_environment_v1.json", runtime_environment())

    implementation_files = [
        PROJECT_ROOT / "src" / "dolphin_behavior_mil" / "modeling_data.py",
        PROJECT_ROOT / "src" / "dolphin_behavior_mil" / "classical.py",
        PROJECT_ROOT / "src" / "dolphin_behavior_mil" / "mil_models.py",
        PROJECT_ROOT / "src" / "dolphin_behavior_mil" / "training.py",
        PROJECT_ROOT / "src" / "dolphin_behavior_mil" / "experiment.py",
        PROJECT_ROOT / "scripts" / "build_model_registry_v1.py",
    ]
    manifest = {
        "implementation_id": "modeling_implementation_v1",
        "protocol_id": parsed["modeling"]["protocol"]["protocol_id"],
        "source_dataset_version": parsed["modeling"]["protocol"]["source_dataset_version"],
        "configuration_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in configs.values()
        },
        "implementation_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in implementation_files
        },
        "registry_sha256": sha256_file(output_dir / "model_registry_v1.json"),
        "contract_checks_sha256": sha256_file(
            output_dir / "modeling_contract_checks_v1.json"
        ),
        "claims": {
            "per_whistle_behavior_ground_truth_used": False,
            "outer_test_used_during_implementation_smoke_test": False,
            "smoke_metric_is_scientific_result": False,
            "pretrained_embeddings_complete": False,
        },
    }
    atomic_write_json(output_dir / "implementation_manifest_v1.json", manifest)
    print(f"registered_models={len(registry)}")
    print(f"contract_checks={contract['passed']}/{contract['passed'] + contract['failed']}")
    print(output_dir / "implementation_manifest_v1.json")


if __name__ == "__main__":
    main()
