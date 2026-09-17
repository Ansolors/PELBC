"""项目路径解析与原始数据只读检查。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = EXPERIMENT_ROOT.parent


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    encounters: Path
    whistle_clips: Path
    environment_info: Path
    whistle_detector: Path
    manifests: Path
    processed: Path
    results: Path
    figures: Path
    manuscript: Path


def _resolve(base: Path, value: str) -> Path:
    return (base / value).resolve()


def load_paths(config_path: Path | None = None) -> ProjectPaths:
    """读取TOML配置，并将所有路径解析为绝对路径。"""

    config_path = config_path or EXPERIMENT_ROOT / "configs" / "paths.toml"
    if not config_path.is_file():
        raise FileNotFoundError(
            "Create a local configs/paths.toml from configs/paths.example.toml; "
            "keep private paths and data outside the public repository."
        )
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    base = config_path.parent

    return ProjectPaths(
        project_root=PROJECT_ROOT,
        encounters=_resolve(base, config["raw"]["encounters"]),
        whistle_clips=_resolve(base, config["raw"]["whistle_clips"]),
        environment_info=_resolve(base, config["raw"]["environment_info"]),
        whistle_detector=_resolve(base, config["related"]["whistle_detector"]),
        manifests=_resolve(base, config["project"]["manifests"]),
        processed=_resolve(base, config["project"]["processed"]),
        results=_resolve(base, config["project"]["results"]),
        figures=_resolve(base, config["project"]["figures"]),
        manuscript=_resolve(base, config["project"]["manuscript"]),
    )


def validate_paths(paths: ProjectPaths) -> list[str]:
    """返回路径配置问题；空列表表示检查通过。"""

    problems: list[str] = []
    raw_paths = {
        "encounters": paths.encounters,
        "whistle_clips": paths.whistle_clips,
        "environment_info": paths.environment_info,
    }
    for name, path in raw_paths.items():
        if not path.is_dir():
            problems.append(f"原始数据目录不存在: {name}={path}")
        if paths.project_root == path or paths.project_root in path.parents:
            problems.append(f"原始数据目录不得位于项目目录内: {name}={path}")

    for name in ("manifests", "processed", "results", "figures", "manuscript"):
        path = getattr(paths, name)
        if paths.project_root != path and paths.project_root not in path.parents:
            problems.append(f"派生输出必须位于项目目录内: {name}={path}")
    return problems
