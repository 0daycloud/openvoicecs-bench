"""Tests for OpenVoiceCS public/sealed split manifests."""

from __future__ import annotations

import json

from src.evaluation.benchmark.openvoicecs import OpenVoiceCSBench, build_release_audit, load_audio_manifest
from src.evaluation.benchmark.splits import (
    build_split_commitments,
    build_split_commitments_file,
    load_split_manifest,
    split_manifest_stats,
    validate_split_commitments,
    validate_split_commitments_file,
    validate_split_manifest,
    validate_split_manifest_file,
)


def test_split_manifest_validates_against_seed_release():
    scenario_ids = {scenario["id"] for scenario in OpenVoiceCSBench.load().scenarios}
    audio_ids = {variant["id"] for variant in load_audio_manifest()}

    manifest = load_split_manifest()
    stats = split_manifest_stats(
        manifest,
        scenario_ids=scenario_ids,
        audio_variant_ids=audio_ids,
    )

    assert validate_split_manifest_file(
        scenario_ids=scenario_ids,
        audio_variant_ids=audio_ids,
    ) == []
    assert stats["scenario_coverage"] == 1.0
    assert stats["audio_variant_coverage"] == 1.0
    assert stats["splits"]["public_dev"]["num_scenarios"] == 80
    assert stats["splits"]["sealed_test"]["num_scenarios"] == 121
    assert (
        stats["splits"]["public_dev"]["num_scenarios"]
        + stats["splits"]["sealed_test"]["num_scenarios"]
        == len(scenario_ids)
    )


def test_split_manifest_rejects_overlap_unknown_and_missing_required_split():
    manifest = {
        "name": "bad",
        "version": "0.1.0",
        "splits": {
            "public_dev": {
                "scenario_ids": ["known", "known", "also-sealed", "unknown"],
                "audio_variant_ids": ["audio-known"],
            },
            "sealed_test": {
                "scenario_ids": ["also-sealed"],
                "audio_variant_ids": ["audio-known"],
            },
        },
    }

    messages = {
        (issue.item_id, issue.message)
        for issue in validate_split_manifest(
            manifest,
            scenario_ids={"known", "also-sealed"},
            audio_variant_ids={"audio-known"},
        )
    }

    assert ("known", "duplicate scenario id in split") in messages
    assert ("unknown", "unknown scenario id") in messages
    assert ("also-sealed", "scenario id also assigned to split public_dev") in messages
    assert ("audio-known", "audio variant id also assigned to split public_dev") in messages


def test_release_audit_includes_split_manifest(tmp_path):
    audit = build_release_audit()
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    assert audit["validation"]["passed"] is True
    assert audit["release_gates"]["has_split_manifest"] is True
    assert audit["release_gates"]["all_scenarios_assigned_to_split"] is True
    assert audit["release_gates"]["all_audio_variants_assigned_to_split"] is True
    assert audit["split_manifest_stats"]["scenario_coverage"] == 1.0
    assert len(audit["files"]["split_manifest"]["sha256"]) == 64


def test_split_commitments_are_deterministic_and_validate(tmp_path):
    output_path = tmp_path / "split_commitments.json"

    first = build_split_commitments_file(
        scenario_path="data/openvoicecs/scenarios_v0.1.json",
        split_path="data/openvoicecs/splits_v0.1.json",
        audio_manifest_path="data/openvoicecs/audio_manifest_v0.1.json",
        output_path=output_path,
    )
    second = build_split_commitments_file(
        scenario_path="data/openvoicecs/scenarios_v0.1.json",
        split_path="data/openvoicecs/splits_v0.1.json",
        audio_manifest_path="data/openvoicecs/audio_manifest_v0.1.json",
    )

    assert first == second
    assert len(first["root_hash"]) == 64
    assert first["splits"]["public_dev"]["num_scenarios"] == 80
    assert first["splits"]["sealed_test"]["num_scenarios"] == 121
    assert first["splits"]["public_dev"]["scenario_commitments"][0]["id"]
    assert validate_split_commitments_file(
        output_path,
        scenario_path="data/openvoicecs/scenarios_v0.1.json",
        split_path="data/openvoicecs/splits_v0.1.json",
        audio_manifest_path="data/openvoicecs/audio_manifest_v0.1.json",
    ) == []


def test_split_commitments_hide_sealed_ids_by_default():
    suite = {
        "version": "test",
        "scenarios": [
            {"id": "public-001", "domain": "retail"},
            {"id": "sealed-001", "domain": "travel"},
        ],
    }
    split_manifest = {
        "version": "test",
        "splits": {
            "public_dev": {"scenario_ids": ["public-001"], "audio_variant_ids": []},
            "sealed_test": {"scenario_ids": ["sealed-001"], "audio_variant_ids": []},
        },
    }

    commitments = build_split_commitments(
        suite=suite,
        split_manifest=split_manifest,
    )

    assert commitments["splits"]["public_dev"]["scenario_commitments"][0]["id"] == "public-001"
    assert "id" not in commitments["splits"]["sealed_test"]["scenario_commitments"][0]
    assert validate_split_commitments(
        commitments,
        suite=suite,
        split_manifest=split_manifest,
    ) == []


def test_split_commitments_detect_tampering():
    suite = {
        "version": "test",
        "scenarios": [{"id": "public-001", "domain": "retail"}],
    }
    split_manifest = {
        "version": "test",
        "splits": {
            "public_dev": {"scenario_ids": ["public-001"], "audio_variant_ids": []},
            "sealed_test": {"scenario_ids": [], "audio_variant_ids": []},
        },
    }
    commitments = build_split_commitments(
        suite=suite,
        split_manifest=split_manifest,
    )
    commitments["splits"]["public_dev"]["scenario_commitments"][0]["sha256"] = "0" * 64

    messages = {
        (issue.path, issue.message)
        for issue in validate_split_commitments(
            commitments,
            suite=suite,
            split_manifest=split_manifest,
        )
    }

    assert (
        "splits.public_dev.scenario_commitments",
        "commitments do not match release files",
    ) in messages
