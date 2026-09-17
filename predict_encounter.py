#!/usr/bin/env python3
"""Repository-local command-line entry point."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pelbc.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
