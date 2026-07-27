"""Tests for OpenVoiceCS release changelog and errata metadata."""

from __future__ import annotations

from src.evaluation.benchmark.changelog import (
    changelog_stats,
    load_changelog,
    validate_changelog,
    validate_changelog_file,
)
from src.evaluation.benchmark.openvoicecs import OpenVoiceCSBench, load_audio_manifest


def test_seed_changelog_covers_seed_release_items():
    bench = OpenVoiceCSBench.load()
    scenario_ids = {scenario["id"] for scenario in bench.scenarios}
    audio_variant_ids = {variant["id"] for variant in load_audio_manifest()}

    changelog = load_changelog()
    issues = validate_changelog_file(
        scenario_ids=scenario_ids,
        audio_variant_ids=audio_variant_ids,
        benchmark_version=bench.version,
    )
    stats = changelog_stats(
        changelog,
        scenario_ids=scenario_ids,
        audio_variant_ids=audio_variant_ids,
    )

    assert issues == []
    assert stats["present"] is True
    assert stats["num_entries"] == len(changelog["entries"])
    assert stats["entry_types"]["release"] == 1
    assert set(stats["entry_types"]) <= {"release", "scenario_added", "erratum"}
    assert stats["scenario_change_coverage"] == 1.0
    assert stats["audio_variant_change_coverage"] == 1.0
    assert stats["num_open_errata"] == 0


def test_changelog_rejects_unknown_ids_bad_types_and_duplicate_entries():
    changelog = {
        "name": "bad changelog",
        "version": "0.1.0",
        "benchmark_version": "0.2.0",
        "entries": [
            {
                "id": "dup",
                "type": "release",
                "date": "2026-06-11",
                "summary": "first",
                "compatibility": "initial_release",
                "scenario_ids": ["known", "unknown"],
                "audio_variant_ids": ["audio-known"],
            },
            {
                "id": "dup",
                "type": "bad_type",
                "date": "",
                "summary": "",
                "compatibility": "bad_compat",
                "scenario_ids": "known",
            },
        ],
        "errata": [
            {
                "id": "err-1",
                "summary": "",
                "status": "triaged",
                "severity": "severe",
                "scenario_ids": ["unknown"],
                "audio_variant_ids": ["audio-unknown"],
            }
        ],
    }

    messages = {
        (issue.item_id, issue.path, issue.message)
        for issue in validate_changelog(
            changelog,
            scenario_ids={"known"},
            audio_variant_ids={"audio-known"},
            benchmark_version="0.1.0",
        )
    }

    assert (
        "<changelog>",
        "benchmark_version",
        "must match scenario suite version",
    ) in messages
    assert ("unknown", "entries[0].scenario_ids[1]", "unknown scenario id") in messages
    assert ("dup", "entries[1].id", "duplicate id") in messages
    assert ("dup", "entries[1].type", "must be a supported change type") in messages
    assert (
        "dup",
        "entries[1].compatibility",
        "must be a supported compatibility class",
    ) in messages
    assert ("dup", "entries[1].scenario_ids", "must be a list") in messages
    assert ("err-1", "errata[0].status", "must be open, resolved, or wontfix") in messages
    assert (
        "err-1",
        "errata[0].severity",
        "must be low, medium, high, or critical",
    ) in messages
