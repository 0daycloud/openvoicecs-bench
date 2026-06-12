#!/usr/bin/env python3
"""Refresh SHA-256 and byte-size pins in OpenVoiceCS JSON evidence files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def _pin_entry(entry: dict[str, Any], *, base_dir: Path) -> bool:
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return False
    file_path = Path(raw_path)
    if not file_path.is_absolute():
        file_path = base_dir / file_path
    if not file_path.exists() or not file_path.is_file():
        return False
    data = file_path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    changed = entry.get("sha256") != sha256 or entry.get("bytes") != len(data)
    entry["sha256"] = sha256
    entry["bytes"] = len(data)
    return changed


def _walk(value: Any, *, base_dir: Path) -> int:
    changed = 0
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"}.issubset(value):
            changed += 1 if _pin_entry(value, base_dir=base_dir) else 0
        for child in value.values():
            changed += _walk(child, base_dir=base_dir)
    elif isinstance(value, list):
        for child in value:
            changed += _walk(child, base_dir=base_dir)
    return changed


def repin_file(path: Path, *, base_dir: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = _walk(data, base_dir=base_dir)
    if changed:
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="JSON evidence files to repin")
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory for resolving relative pinned paths.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    total = 0
    for raw_path in args.files:
        path = Path(raw_path)
        changed = repin_file(path, base_dir=base_dir)
        total += changed
        print(f"{path}: refreshed {changed} file entries")
    print(f"Total refreshed entries: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
