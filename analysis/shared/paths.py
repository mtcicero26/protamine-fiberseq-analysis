"""Portable paths shared by analysis scripts.

Primary data and reference resources are intentionally external to this
repository.  Their locations can be supplied with environment variables; the
defaults describe the staging layout used in the workflow documentation.
"""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("PROTAMINE_DATA_ROOT", REPOSITORY_ROOT / "data" / "primary"))
REFERENCE_ROOT = Path(
    os.environ.get("PROTAMINE_REFERENCE_ROOT", REPOSITORY_ROOT / "data" / "reference")
)
CACHE_ROOT = Path(os.environ.get("PROTAMINE_CACHE_ROOT", REPOSITORY_ROOT / "data" / "cache"))
RESULTS_ROOT = Path(os.environ.get("PROTAMINE_OUTPUT_ROOT", REPOSITORY_ROOT / "results"))
SOURCE_DATA_ROOT = REPOSITORY_ROOT / "data" / "source"
MODEL_OUTPUT_ROOT = Path(
    os.environ.get(
        "PROTAMINE_MODEL_OUTPUT_ROOT",
        REPOSITORY_ROOT / "data" / "processed" / "fiberformer",
    )
)


def output_directory(*parts: str) -> Path:
    """Return and create a directory below the configured results root."""
    path = RESULTS_ROOT.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
