"""Command-line interface for one encounter-level prediction."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from typing import Any

from .errors import PELBCError
from .predictor import EncounterPredictor
from .version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pelbc-predict",
        description=(
            "Score Feeding, Travelling, Milling, and Socializing for one encounter "
            "represented by pre-extracted whistle WAV clips."
        ),
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="directory containing whistle clips from one encounter",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output .json or .csv path; JSON is printed to stdout when omitted",
    )
    parser.add_argument(
        "--encounter-id",
        help="encounter identifier stored in the output (defaults to the directory name)",
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="private local portable model JSON; trained parameters are not distributed",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="search for WAV files recursively below --input-dir",
    )
    parser.add_argument(
        "--no-clip-details",
        action="store_true",
        help="omit per-clip hashes and quality summaries from JSON output",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress progress and the human-readable score summary",
    )
    parser.add_argument("--version", action="version", version=f"PELBC {__version__}")
    return parser


def _json_text(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2) + "\n"


def _csv_text(result: dict[str, Any]) -> str:
    buffer = io.StringIO(newline="")
    fieldnames = [
        "encounter_id",
        "rank",
        "behavior",
        "score",
        "whistle_clip_count",
        "quality_status",
        "warning_codes",
        "model_id",
        "model_sha256",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    warning_codes = ";".join(
        str(row["code"]) for row in result["quality_control"]["warnings"]
    )
    for row in result["ranked_behaviors"]:
        writer.writerow(
            {
                "encounter_id": result["encounter_id"],
                "rank": row["rank"],
                "behavior": row["label"],
                "score": format(float(row["score"]), ".12g"),
                "whistle_clip_count": result["input_summary"]["whistle_clip_count"],
                "quality_status": result["quality_control"]["status"],
                "warning_codes": warning_codes,
                "model_id": result["model"]["model_id"],
                "model_sha256": result["model"]["artifact_sha256"],
            }
        )
    return buffer.getvalue()


def _atomic_write(path: Path, text: str) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _render_output(result: dict[str, Any], output: Path | None) -> str | None:
    if output is None:
        return _json_text(result)
    suffix = output.suffix.casefold()
    if suffix == ".json":
        text = _json_text(result)
    elif suffix == ".csv":
        text = _csv_text(result)
    else:
        raise PELBCError("--output must end in .json or .csv")
    _atomic_write(output, text)
    return None


def _print_summary(result: dict[str, Any], output: Path | None) -> None:
    print(
        f"Encounter {result['encounter_id']}: "
        f"{result['input_summary']['whistle_clip_count']} whistle clip(s)",
        file=sys.stderr,
    )
    for row in result["ranked_behaviors"]:
        print(f"  {row['rank']}. {row['label']}: {row['score']:.4f}", file=sys.stderr)
    quality = result["quality_control"]
    print(
        f"Quality control: {quality['status']} ({quality['warning_count']} warning(s))",
        file=sys.stderr,
    )
    if output is not None:
        print(f"Saved: {output.expanduser().resolve()}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        predictor = EncounterPredictor.from_model_path(args.model)

        def progress(index: int, total: int, path: Path) -> None:
            if not args.quiet:
                print(f"[{index}/{total}] {path.name}", file=sys.stderr)

        result = predictor.predict_directory(
            args.input_dir,
            encounter_id=args.encounter_id,
            recursive=args.recursive,
            include_clip_details=not args.no_clip_details,
            progress=progress,
        )
        stdout_text = _render_output(result, args.output)
        if stdout_text is not None:
            sys.stdout.write(stdout_text)
        if not args.quiet:
            _print_summary(result, args.output)
        return 0
    except (PELBCError, OSError, ValueError) as exc:
        print(f"PELBC error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
