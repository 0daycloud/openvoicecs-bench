"""Tests for OpenVoiceCS coverage planning."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.benchmark.coverage import build_coverage_plan


def test_public_beta_coverage_plan_reports_current_release_coverage():
    plan = build_coverage_plan(profile="public_beta")

    assert plan["passed"] is True
    assert plan["gaps"]["total"]["current"] == 201
    assert plan["gaps"]["total"]["target"] == 50
    assert plan["gaps"]["total"]["needed"] == 0
    assert plan["gaps"]["tracks"]["audio_to_action"]["needed"] == 0
    assert plan["gaps"]["tracks"]["end_to_end_voice"]["needed"] == 0
    assert plan["gaps"]["splits"]["sealed_test"]["needed"] == 0
    assert plan["gaps"]["audio_variants"]["total"]["current"] == 120
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
    targets = {
        "name": "test targets",
        "version": "test",
        "profiles": {
            "one_more": {
                "min_scenarios": 202,
                "domains": {
                    "retail": 34,
                },
                "tracks": {
                    "text_to_action": 62,
                },
                "difficulty": {"medium": 82},
                "splits": {"sealed_test": 122},
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
    targets = {
        "name": "test targets",
        "version": "test",
        "profiles": {
            "one_domain_short": {
                "min_scenarios": 201,
                "domains": {
                    "travel": 35,
                },
                "tracks": {
                    "text_to_action": 61,
                },
                "difficulty": {"medium": 81},
                "splits": {"sealed_test": 121},
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
