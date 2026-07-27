"""Tests for OpenVoiceCS coverage planning."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.benchmark.coverage import build_coverage_plan
from src.evaluation.benchmark.openvoicecs import OpenVoiceCSBench, load_audio_manifest


def _current_counts() -> dict[str, dict[str, int]]:
    """Read the live corpus counts the planner is expected to reproduce."""
    gaps = build_coverage_plan(profile="public_beta")["gaps"]
    return {
        dimension: {key: entry["current"] for key, entry in gaps[dimension].items()}
        for dimension in ("domains", "tracks", "difficulty", "splits")
    }


def test_public_beta_coverage_plan_reports_current_release_coverage():
    plan = build_coverage_plan(profile="public_beta")

    assert plan["passed"] is True
    assert plan["gaps"]["total"]["current"] == len(OpenVoiceCSBench.load().scenarios)
    assert plan["gaps"]["total"]["target"] == 50
    assert plan["gaps"]["total"]["needed"] == 0
    assert plan["gaps"]["tracks"]["audio_to_action"]["needed"] == 0
    assert plan["gaps"]["tracks"]["end_to_end_voice"]["needed"] == 0
    assert plan["gaps"]["splits"]["sealed_test"]["needed"] == 0
    assert plan["gaps"]["audio_variants"]["total"]["current"] == len(load_audio_manifest())
    assert plan["gaps"]["audio_variants"]["total"]["needed"] == 0
    assert plan["gaps"]["audio_variants"]["splits"]["public_dev"]["needed"] == 0
    assert plan["gaps"]["audio_variants"]["splits"]["sealed_test"]["needed"] == 0
    assert plan["recommended_next_scenarios"] == []


def test_coverage_plan_passes_when_targets_match_seed(tmp_path: Path):
    targets = {
        "name": "test targets",
        "version": "test",
        "profiles": {
            "seed_exact": {
                "min_scenarios": 13,
                "domains": {
                    "retail": 2,
                    "travel": 2,
                    "telecom": 2,
                    "healthcare_admin": 2,
                    "fintech_sandbox": 2,
                    "saas_support": 2,
                    "utility_support": 1,
                },
                "tracks": {
                    "text_to_action": 6,
                    "adversarial_compliance": 4,
                    "audio_to_action": 1,
                    "robustness": 1,
                    "end_to_end_voice": 1,
                },
                "difficulty": {"easy": 1, "medium": 4, "hard": 8},
                "splits": {"public_dev": 13, "sealed_test": 0},
                "audio_variants": {
                    "total": 10,
                    "splits": {"public_dev": 10, "sealed_test": 0},
                },
            }
        },
    }
    target_path = tmp_path / "targets.json"
    target_path.write_text(json.dumps(targets), encoding="utf-8")

    plan = build_coverage_plan(target_path=target_path, profile="seed_exact")

    assert plan["passed"] is True
    assert plan["recommended_next_scenarios"] == []


def test_coverage_plan_caps_recommendations_to_total_gap(tmp_path: Path):
    current = _current_counts()
    targets = {
        "name": "test targets",
        "version": "test",
        "profiles": {
            "one_more": {
                "min_scenarios": sum(current["splits"].values()) + 1,
                "domains": {"retail": current["domains"]["retail"] + 1},
                "tracks": {"text_to_action": current["tracks"]["text_to_action"] + 1},
                "difficulty": {"medium": current["difficulty"]["medium"] + 1},
                "splits": {"sealed_test": current["splits"]["sealed_test"] + 1},
            }
        },
    }
    target_path = tmp_path / "targets.json"
    target_path.write_text(json.dumps(targets), encoding="utf-8")

    plan = build_coverage_plan(target_path=target_path, profile="one_more")

    assert plan["gaps"]["total"]["needed"] == 1
    assert len(plan["recommended_next_scenarios"]) == 1
    assert plan["recommended_next_scenarios"][0]["track"] is not None
    assert plan["recommended_next_scenarios"][0]["split"] is not None


def test_coverage_plan_recommends_subgroup_fill_when_total_is_met(tmp_path: Path):
    current = _current_counts()
    targets = {
        "name": "test targets",
        "version": "test",
        "profiles": {
            "one_domain_short": {
                "min_scenarios": sum(current["splits"].values()),
                "domains": {"travel": current["domains"]["travel"] + 1},
                "tracks": {"text_to_action": current["tracks"]["text_to_action"]},
                "difficulty": {"medium": current["difficulty"]["medium"]},
                "splits": {"sealed_test": current["splits"]["sealed_test"]},
            }
        },
    }
    target_path = tmp_path / "targets.json"
    target_path.write_text(json.dumps(targets), encoding="utf-8")

    plan = build_coverage_plan(target_path=target_path, profile="one_domain_short")

    assert plan["gaps"]["total"]["needed"] == 0
    assert plan["gaps"]["domains"]["travel"]["needed"] == 1
    assert len(plan["recommended_next_scenarios"]) == 1
    assert plan["recommended_next_scenarios"][0]["domain"] == "travel"
    assert plan["recommended_next_scenarios"][0]["track"] == "text_to_action"
