"""Tests for OpenVoiceCS external-system registry evidence."""

from __future__ import annotations

from copy import deepcopy

from src.evaluation.benchmark.external_systems import (
    external_systems_stats,
    load_external_systems_registry,
    validate_external_systems_registry,
    validate_external_systems_registry_file,
)


def test_external_systems_registry_validates():
    registry = load_external_systems_registry()

    assert validate_external_systems_registry(registry) == []
    assert validate_external_systems_registry_file() == []
    assert external_systems_stats(registry)["reference_fixtures"] == 2
    assert external_systems_stats(registry)["official_systems"] == 0


def test_external_systems_registry_detects_tampered_hash():
    registry = load_external_systems_registry()
    registry["systems"][0]["report"]["sha256"] = "0" * 64

    messages = {
        (issue.path, issue.message)
        for issue in validate_external_systems_registry(registry)
    }

    assert ("systems[0].report.sha256", "does not match file contents") in messages


def test_external_systems_registry_rejects_reference_fixture_as_official():
    registry = load_external_systems_registry()
    system = deepcopy(registry["systems"][0])
    system["id"] = "bad-official-reference"
    system["status"] = "official"
    system["official_leaderboard_eligible"] = True
    registry["systems"] = [system]

    messages = {
        (issue.path, issue.message)
        for issue in validate_external_systems_registry(registry)
    }

    assert ("systems[0].provider", "official systems must not use reference provider") in messages
    assert (
        "systems[0].judge_evidence.annotation_mode",
        "official systems cannot use reference_fixture annotations",
    ) in messages
