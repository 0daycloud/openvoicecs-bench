"""Tests for OpenVoiceCS scenario authoring utilities."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from src.evaluation.benchmark.openvoicecs import OpenVoiceCSBench, validate_scenarios
from src.evaluation.benchmark.scenario_authoring import (
    add_scenarios_to_release_files,
    next_scenario_id,
    scaffold_scenario_drafts,
)


def test_next_scenario_id_increments_matching_prefix_only():
    existing = {
        "retail-refund-damaged-item-001",
        "retail-refund-damaged-item-003",
        "retail-other-999",
    }

    assert (
        next_scenario_id(existing, domain="retail", slug="refund damaged item")
        == "retail-refund-damaged-item-004"
    )


def test_add_scenarios_updates_suite_split_and_provenance(tmp_path: Path):
    bench = OpenVoiceCSBench.load()
    scenario = deepcopy(bench.scenarios[0])
    scenario["id"] = "retail-refund-damaged-item-999"
    scenario["customer_goal"] = "Customer wants a refund for a second synthetic damaged item."
    draft_path = tmp_path / "draft.json"
    draft_path.write_text(json.dumps({"scenarios": [scenario]}), encoding="utf-8")

    result = add_scenarios_to_release_files(
        draft_path=draft_path,
        scenario_path="data/openvoicecs/scenarios_v0.1.json",
        split_path="data/openvoicecs/splits_v0.1.json",
        provenance_path="data/openvoicecs/provenance_v0.1.json",
        audio_manifest_path="data/openvoicecs/audio_manifest_v0.1.json",
        output_scenario_path=tmp_path / "scenarios.json",
        output_split_path=tmp_path / "splits.json",
        output_provenance_path=tmp_path / "provenance.json",
    )

    assert result["issues"] == []
    assert result["added_ids"] == ["retail-refund-damaged-item-999"]
    scenarios = json.loads((tmp_path / "scenarios.json").read_text(encoding="utf-8"))
    splits = json.loads((tmp_path / "splits.json").read_text(encoding="utf-8"))
    provenance = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert len(scenarios["scenarios"]) == len(bench.scenarios) + 1
    assert "retail-refund-damaged-item-999" in splits["splits"]["public_dev"]["scenario_ids"]
    assert "retail-refund-damaged-item-999" in provenance["scenarios"]
    assert provenance["scenarios"]["retail-refund-damaged-item-999"]["review_status"] == "draft"


def test_add_scenarios_rejects_invalid_draft_without_writing(tmp_path: Path):
    scenario = deepcopy(OpenVoiceCSBench.load().scenarios[0])
    scenario["id"] = "retail-refund-damaged-item-001"
    scenario["oracle"]["expected_tool_calls"][0]["arguments"]["account_id"] = "wrong"
    draft_path = tmp_path / "bad_draft.json"
    draft_path.write_text(json.dumps({"scenarios": [scenario]}), encoding="utf-8")

    result = add_scenarios_to_release_files(
        draft_path=draft_path,
        scenario_path="data/openvoicecs/scenarios_v0.1.json",
        split_path="data/openvoicecs/splits_v0.1.json",
        provenance_path="data/openvoicecs/provenance_v0.1.json",
        audio_manifest_path="data/openvoicecs/audio_manifest_v0.1.json",
        output_scenario_path=tmp_path / "scenarios.json",
        output_split_path=tmp_path / "splits.json",
        output_provenance_path=tmp_path / "provenance.json",
    )

    messages = {(issue.item_id, issue.path, issue.message) for issue in result["issues"]}

    assert ("retail-refund-damaged-item-001", "id", "duplicate scenario id") in messages
    assert (
        "retail-refund-damaged-item-001",
        "oracle.expected_tool_calls",
        "oracle calls do not replay cleanly",
    ) in messages
    assert not (tmp_path / "scenarios.json").exists()


def test_scaffold_scenario_drafts_uses_coverage_recommendations(tmp_path: Path):
    target_path = _extra_scenario_target_path(tmp_path)
    draft_suite = scaffold_scenario_drafts(
        target_path=target_path,
        profile="extra_scenarios",
        count=3,
    )

    assert draft_suite["draft_status"] == "incomplete_scaffold"
    assert draft_suite["num_scenarios"] == 3
    scenario_ids = [scenario["id"] for scenario in draft_suite["scenarios"]]
    assert len(set(scenario_ids)) == 3
    assert all("-draft-" in scenario_id for scenario_id in scenario_ids)
    assert draft_suite["scenarios"][0]["draft_metadata"]["status"] == "incomplete_scaffold"
    assert {
        scenario["draft_metadata"]["coverage_split"]
        for scenario in draft_suite["scenarios"]
    } <= {"public_dev", "sealed_test"}


def test_scaffold_scenario_drafts_do_not_validate_as_release_ready(tmp_path: Path):
    draft_suite = scaffold_scenario_drafts(
        target_path=_extra_scenario_target_path(tmp_path),
        profile="extra_scenarios",
        count=1,
    )

    issues = validate_scenarios(draft_suite["scenarios"])

    assert any(issue.path == "oracle.expected_state" for issue in issues)


def _extra_scenario_target_path(tmp_path: Path) -> Path:
    target_path = tmp_path / "openvoicecs_extra_scenario_targets.json"
    target_path.write_text(
        json.dumps({
            "name": "test targets",
            "version": "test",
            "profiles": {
                "extra_scenarios": {
                    "min_scenarios": 210,
                    "domains": {
                        "retail": 40,
                    },
                    "tracks": {
                        "text_to_action": 70,
                    },
                    "difficulty": {"medium": 90},
                    "splits": {"sealed_test": 130},
                }
            },
        }),
        encoding="utf-8",
    )
    return target_path
