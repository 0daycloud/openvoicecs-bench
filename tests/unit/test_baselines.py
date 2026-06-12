"""Tests for OpenVoiceCS reference baseline packages."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.benchmark.baselines import (
    build_reference_baselines,
    validate_reference_baselines,
    validate_reference_baselines_file,
)
from src.evaluation.benchmark.openvoicecs import load_audio_manifest


def test_build_reference_baselines_writes_reports_and_valid_manifest(tmp_path: Path):
    output_dir = tmp_path / "baselines"
    manifest_path = output_dir / "reference_baselines.json"

    manifest = build_reference_baselines(
        output_dir=output_dir,
        manifest_path=manifest_path,
        trials=2,
    )

    baseline_ids = {baseline["id"] for baseline in manifest["baselines"]}
    assert baseline_ids == {
        "oracle_text",
        "noop_text",
        "oracle_audio_manifest",
        "noop_audio_manifest",
    }
    assert validate_reference_baselines(manifest) == []
    assert validate_reference_baselines_file(manifest_path) == []
    assert manifest["baselines"][0]["report"]["sha256"]

    by_id = {baseline["id"]: baseline for baseline in manifest["baselines"]}
    assert by_id["oracle_text"]["expected"]["overall_score"] == 100.0
    assert by_id["oracle_text"]["expected"]["pass_k"] == 1.0
    assert by_id["noop_text"]["expected"]["pass_at_k"] == 0.0
    assert by_id["oracle_audio_manifest"]["expected"]["num_scenarios"] == len(load_audio_manifest())


def test_validate_reference_baselines_detects_tampered_report(tmp_path: Path):
    output_dir = tmp_path / "baselines"
    manifest_path = output_dir / "reference_baselines.json"
    manifest = build_reference_baselines(
        output_dir=output_dir,
        manifest_path=manifest_path,
        trials=1,
    )
    report_path = Path(manifest["baselines"][0]["report"]["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["overall_score"] = 99.0
    report_path.write_text(json.dumps(report), encoding="utf-8")

    messages = {
        (issue.path, issue.message)
        for issue in validate_reference_baselines_file(manifest_path)
    }

    assert ("baselines[0].report.sha256", "does not match file contents") in messages
    assert ("baselines[0].expected.overall_score", "does not match report") in messages
    assert ("baselines[0].report.overall_score", "oracle baseline must score 100") in messages


def test_validate_reference_baselines_rejects_missing_required_baseline():
    manifest = {
        "name": "OpenVoiceCS-Bench Reference Baselines",
        "version": "0.1.0",
        "benchmark_version": "0.1.0",
        "scenario_file": {
            "path": "data/openvoicecs/scenarios_v0.1.json",
            "sha256": "a" * 64,
            "bytes": 1,
        },
        "baselines": [],
    }

    messages = {
        (issue.item_id, issue.path, issue.message)
        for issue in validate_reference_baselines(manifest)
    }

    assert ("<baselines>", "baselines", "must be a non-empty list") in messages
