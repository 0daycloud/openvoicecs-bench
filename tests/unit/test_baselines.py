"""Tests for OpenVoiceCS reference baseline packages."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.benchmark.baselines import (
    build_reference_baselines,
    validate_reference_baselines,
    validate_reference_baselines_file,
)
from src.evaluation.benchmark.openvoicecs import OpenVoiceCSBench, load_audio_manifest, no_op_agent

# Reference baselines must stay generatable and deterministic without a judge
# API key: build_reference_baselines forces grounding_mode="legacy", so the
# hybrid semantic fallback (which needs OPENAI_API_KEY/OPENROUTER_API_KEY)
# never triggers here even though it's the default for normal scoring.
_JUDGE_KEY_ENV_VARS = ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_KEY", "OPEN_ROUTER_API_KEY")


def _without_judge_keys(monkeypatch):
    for var in _JUDGE_KEY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


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

    # _file_entry() paths must be platform-independent (forward-slash) so the
    # manifest is byte-identical whether generated on Windows or POSIX.
    manifest_paths = [manifest["scenario_file"]["path"], manifest["audio_manifest_file"]["path"]]
    manifest_paths += [baseline["report"]["path"] for baseline in manifest["baselines"]]
    for path in manifest_paths:
        assert "\\" not in path, path
        assert "/" in path, path

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


def test_build_reference_baselines_measures_noop_without_a_judge_key(monkeypatch, tmp_path: Path):
    _without_judge_keys(monkeypatch)
    output_dir = tmp_path / "baselines"
    manifest_path = output_dir / "reference_baselines.json"

    manifest = build_reference_baselines(
        output_dir=output_dir,
        manifest_path=manifest_path,
        trials=1,
    )

    by_id = {baseline["id"]: baseline for baseline in manifest["baselines"]}
    report = json.loads(Path(by_id["noop_text"]["report"]["path"]).read_text(encoding="utf-8"))

    assert report["num_measured_scenarios"] == report["num_scenarios"]
    infrastructure_trials = [
        trial
        for scenario_result in report["results"]
        for trial in scenario_result["trials"]
        if trial.get("error_class") == "infrastructure"
    ]
    assert infrastructure_trials == []


def test_build_reference_baselines_legacy_trials_omit_grounding_mode_metadata(monkeypatch, tmp_path: Path):
    # grounding_mode is only attached for non-legacy modes (see
    # check_factual_grounding), so a byte-for-byte reproducible legacy baseline
    # report doesn't carry metadata absent from the pre-hybrid report format.
    _without_judge_keys(monkeypatch)
    output_dir = tmp_path / "baselines"
    manifest_path = output_dir / "reference_baselines.json"

    manifest = build_reference_baselines(
        output_dir=output_dir,
        manifest_path=manifest_path,
        trials=1,
    )

    by_id = {baseline["id"]: baseline for baseline in manifest["baselines"]}
    report = json.loads(Path(by_id["noop_text"]["report"]["path"]).read_text(encoding="utf-8"))

    grounding_checks = [
        trial["grounding_check"]
        for scenario_result in report["results"]
        for trial in scenario_result["trials"]
    ]
    assert grounding_checks
    assert all("grounding_mode" not in check for check in grounding_checks)
    assert all("semantic_fallback" not in check for check in grounding_checks)


def test_build_reference_baselines_report_hashes_are_stable_across_runs(monkeypatch, tmp_path: Path):
    _without_judge_keys(monkeypatch)

    manifest_a = build_reference_baselines(
        output_dir=tmp_path / "run_a",
        manifest_path=tmp_path / "run_a" / "reference_baselines.json",
        trials=1,
    )
    manifest_b = build_reference_baselines(
        output_dir=tmp_path / "run_b",
        manifest_path=tmp_path / "run_b" / "reference_baselines.json",
        trials=1,
    )

    hashes_a = {baseline["id"]: baseline["report"]["sha256"] for baseline in manifest_a["baselines"]}
    hashes_b = {baseline["id"]: baseline["report"]["sha256"] for baseline in manifest_b["baselines"]}
    assert hashes_a == hashes_b


def test_score_agent_default_grounding_mode_stays_hybrid(monkeypatch):
    # baselines.py now forces mode="legacy" explicitly; this pins that
    # score_agent's own default (no grounding_mode argument) is still
    # hybrid, using the same missing-judge-key signal as the baseline
    # test above -- without a key, a hybrid-mode trial on unmatched text
    # must fail via the judge call, not skip it.
    _without_judge_keys(monkeypatch)
    bench = OpenVoiceCSBench.load()

    report = bench.score_agent(no_op_agent, max_scenarios=1, trials=1)

    assert report["results"][0]["num_infrastructure_error_trials"] == 1
    assert report["results"][0]["measured"] is False


def test_build_reference_baselines_writes_lf_only_json(tmp_path: Path):
    # Path.write_text() without newline="\n" translates "\n" to the platform
    # line separator on Windows, producing CRLF reports that diverge from the
    # LF-committed artifacts on every line.
    output_dir = tmp_path / "baselines"
    manifest_path = output_dir / "reference_baselines.json"

    manifest = build_reference_baselines(
        output_dir=output_dir,
        manifest_path=manifest_path,
        trials=1,
    )

    assert b"\r\n" not in manifest_path.read_bytes()
    for baseline in manifest["baselines"]:
        assert b"\r\n" not in Path(baseline["report"]["path"]).read_bytes()


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
