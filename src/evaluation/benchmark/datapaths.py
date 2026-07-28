"""Central resolution of the OpenVoiceCS-Bench data directory.

Every ``DEFAULT_*_PATH`` constant in this package is built from :func:`data_path`
so that installs which are not editable can still locate the corpus.

Resolution order (first hit wins):

1. ``$OPENVOICECS_DATA_DIR`` -- explicit override, absolute or ``~``-relative.
2. ``./data/openvoicecs`` under the current working directory, when it exists.
   This preserves the historical CWD-relative behaviour exactly, including the
   relative path strings that get embedded in generated manifests and reports.
3. ``data/openvoicecs`` beside an installed or checked-out copy of this package,
   located by walking up from this file.
4. ``data/openvoicecs`` relative to the CWD as a last resort, so "file not
   found" errors keep naming the familiar path.

The corpus is ~160 MB, so it ships with the git repository and the release
archives rather than inside the wheel. Point ``OPENVOICECS_DATA_DIR`` at a
checkout or an unpacked release when running from a non-editable install.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Environment variable that overrides data directory discovery.
ENV_VAR = "OPENVOICECS_DATA_DIR"

_RELATIVE_DATA_DIR = Path("data") / "openvoicecs"


def _adjacent_data_dir() -> Path | None:
    """Find ``data/openvoicecs`` beside the installed/checked-out package."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / _RELATIVE_DATA_DIR
        if candidate.is_dir():
            return candidate
    return None


def resolve_data_dir() -> Path:
    """Return the directory that holds the OpenVoiceCS-Bench data files."""
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override).expanduser()
    if _RELATIVE_DATA_DIR.is_dir():
        return _RELATIVE_DATA_DIR
    adjacent = _adjacent_data_dir()
    if adjacent is not None:
        return adjacent
    return _RELATIVE_DATA_DIR


def data_path(*parts: str) -> Path:
    """Return a path inside the resolved data directory."""
    return resolve_data_dir().joinpath(*parts)
