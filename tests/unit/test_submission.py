"""Tests for external OpenVoiceCS submission adapters."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.evaluation.benchmark.submission import (
    DEFAULT_SUBMISSION_INTAKE_PATH,
    build_submission_card,
    build_submission_card_from_file,
    build_submission_template,
    load_submission_intake,
    load_submission_callable,
    score_submission,
    submission_intake_stats,
    validate_submission_card,
    validate_submission_card_file,
    validate_submission_intake,
    validate_submission_intake_file,
    write_submission_template,
)


def test_load_submission_callable_and_score_text_submission():
    spec = "examples/openvoicecs_submission_adapter.py:run"

    fn = load_submission_callable(spec)
    report = score_submission(spec, max_items=2, trials=1, submission_name="example")

    assert callable(fn)
    assert report["overall_score"] == 100.0
    assert report["model_metadata"]["agent"] == "example"
    assert report["model_metadata"]["submission_spec"] == spec
    assert report["operational_metrics"]["avg_cost_usd"] == 0.01


def test_score_submission_audio_mode():
    report = score_submission(
        "examples/openvoicecs_submission_adapter.py:run",
        mode="audio",
        track="adversarial_compliance",
        max_items=1,
        trials=1,
        submission_name="audio_example",
    )

    assert report["overall_score"] == 100.0
    assert report["evaluation_mode"] == "audio_manifest"
    assert report["num_audio_variants"] == 1
    assert report["model_metadata"]["input_modality"] == "audio"


def test_score_submission_records_pinned_system_metadata():
    report = score_submission(
        "examples/openvoicecs_submission_adapter.py:run",
        max_items=1,
        trials=1,
        submission_name="example",
        provider="reference",
        model_id="example-agent-v0.1",
        pricing_profile_id="reference-zero-v0.1",
        pricing_snapshot_date="2026-06-11",
        pipeline_type="cascaded",
    )

    metadata = report["model_metadata"]
    assert metadata["provider"] == "reference"
    assert metadata["model_id"] == "example-agent-v0.1"
    assert metadata["pricing_profile_id"] == "reference-zero-v0.1"
    assert metadata["pipeline_type"] == "cascaded"


def test_submission_card_builds_and_validates_from_report_file(tmp_path: Path):
    report = score_submission(
        "examples/openvoicecs_submission_adapter.py:run",
        max_items=1,
        trials=1,
        submission_name="example",
        provider="reference",
        model_id="example-agent-v0.1",
        pricing_profile_id="reference-zero-v0.1",
        pricing_snapshot_date="2026-06-11",
        pipeline_type="cascaded",
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    card = build_submission_card_from_file(
        report_path,
        submitter_name="Test Submitter",
        organization="OpenVoiceCS",
        training_data_statement="Synthetic benchmark adapter.",
        safety_statement="Oracle adapter for harness testing.",
        limitations=["Not a real deployed system."],
    )
    card_path = tmp_path / "card.json"
    card_path.write_text(json.dumps(card), encoding="utf-8")

    assert card["system"]["model_id"] == "example-agent-v0.1"
    assert card["evaluation"]["report"]["sha256"]
    assert validate_submission_card(card) == []
    assert validate_submission_card_file(card_path) == []


def test_submission_card_validation_rejects_missing_identity_and_bad_scores():
    report = score_submission(
        "examples/openvoicecs_submission_adapter.py:run",
        max_items=1,
        trials=1,
        submission_name="example",
    )
    card = build_submission_card(report)
    card["system"]["submission_spec"] = None
    card["evaluation"]["overall_score"] = 101
    card["disclosures"]["training_data_statement"] = ""

    messages = {
        (issue.path, issue.message)
        for issue in validate_submission_card(card)
    }

    assert ("system.model_id", "must include model_id or submission_spec") in messages
    assert ("evaluation.overall_score", "must be between 0 and 100") in messages
    assert ("disclosures.training_data_statement", "must be a non-empty string") in messages


def test_load_submission_callable_rejects_bad_specs(tmp_path: Path):
    with pytest.raises(ValueError):
        load_submission_callable("missing_separator")

    path = tmp_path / "adapter.py"
    path.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(AttributeError):
        load_submission_callable(f"{path}:run")


def test_write_submission_template_creates_scoreable_adapter(tmp_path: Path):
    path = write_submission_template(tmp_path / "adapter.py", function_name="run_agent")

    template = path.read_text(encoding="utf-8")
    report = score_submission(f"{path}:run_agent", max_items=1, trials=1, submission_name="starter")

    assert "def run_agent" in template
    assert report["model_metadata"]["agent"] == "starter"
    assert report["num_scenarios"] == 1


def test_submission_template_refuses_overwrite_and_bad_function_name(tmp_path: Path):
    path = tmp_path / "adapter.py"
    write_submission_template(path)

    with pytest.raises(FileExistsError):
        write_submission_template(path)
    with pytest.raises(ValueError):
        build_submission_template(function_name="not-valid")


def test_reference_submission_intake_validates():
    envelope = load_submission_intake()
    stats = submission_intake_stats(envelope)

    assert validate_submission_intake_file(DEFAULT_SUBMISSION_INTAKE_PATH) == []
    assert stats["submission_id"] == "oracle_text_realtime_reference_intake"
    assert stats["status"] == "reference_fixture"
    assert stats["official_submission"] is False
    assert stats["required_artifacts_present"] == 7


def test_submission_intake_rejects_tampered_artifact_hash():
    envelope = load_submission_intake()
    envelope["artifacts"]["report"]["sha256"] = "0" * 64

    messages = {
        (issue.path, issue.message)
        for issue in validate_submission_intake(envelope)
    }

    assert ("artifacts.report.sha256", "does not match file contents") in messages


def test_submission_intake_rejects_official_reference_fixture():
    envelope = deepcopy(load_submission_intake())
    envelope["status"] = "official"
    envelope["official_submission"] = True

    messages = {
        (issue.path, issue.message)
        for issue in validate_submission_intake(envelope)
    }

    assert ("review.evidence_level", "official submissions cannot use reference_fixture evidence") in messages
    assert (
        "artifacts.submission_card.disclosures.safety_statement",
        "official submissions must disclose safety policy",
    ) not in messages
