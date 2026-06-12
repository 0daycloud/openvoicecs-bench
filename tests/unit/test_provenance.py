"""Tests for OpenVoiceCS provenance manifests."""

from __future__ import annotations

from src.evaluation.benchmark.openvoicecs import OpenVoiceCSBench, load_audio_manifest
from src.evaluation.benchmark.provenance import (
    load_provenance_manifest,
    provenance_stats,
    validate_provenance_manifest,
    validate_provenance_manifest_file,
)


def test_seed_provenance_manifest_covers_scenarios_and_audio_variants():
    bench = OpenVoiceCSBench.load()
    scenario_ids = {scenario["id"] for scenario in bench.scenarios}
    audio_variant_ids = {variant["id"] for variant in load_audio_manifest()}

    issues = validate_provenance_manifest_file(
        scenario_ids=scenario_ids,
        audio_variant_ids=audio_variant_ids,
    )
    stats = provenance_stats(
        load_provenance_manifest(),
        scenario_ids=scenario_ids,
        audio_variant_ids=audio_variant_ids,
    )

    assert issues == []
    assert stats["scenario_coverage"] == 1.0
    assert stats["audio_variant_coverage"] == 1.0
    assert stats["no_real_customer_data_rate"] == 1.0
    assert stats["low_contamination_risk_rate"] == 1.0
    assert stats["audio_speaker_consent_rate"] == 1.0


def test_provenance_manifest_rejects_unknown_and_missing_entries():
    manifest = {
        "name": "bad provenance",
        "version": "0.1.0",
        "scenarios": {
            "unknown-scenario-001": {
                "source_type": "hand_authored_synthetic",
                "license": "CC-BY-4.0",
                "authoring_method": "synthetic",
                "contains_real_customer_data": False,
                "contamination_risk": "low",
            }
        },
        "audio_variants": {
            "audio-001": {
                "source_type": "unlicensed_call_recording",
                "license": "restricted",
                "speaker_consent": "missing",
                "voice_rights": "unknown",
                "contains_real_customer_data": True,
                "contamination_risk": "high",
            }
        },
    }

    issues = validate_provenance_manifest(
        manifest,
        scenario_ids={"retail-refund-damaged-item-001"},
        audio_variant_ids={"retail-refund-damaged-item-001-clean-us-female"},
    )
    messages = {(issue.item_id, issue.path, issue.message) for issue in issues}

    assert (
        "unknown-scenario-001",
        "scenarios.unknown-scenario-001",
        "unknown scenario id",
    ) in messages
    assert (
        "retail-refund-damaged-item-001",
        "scenarios",
        "missing scenario provenance",
    ) in messages
    assert (
        "audio-001",
        "audio_variants.audio-001",
        "unknown audio variant id",
    ) in messages
    assert (
        "audio-001",
        "audio_variants.audio-001.source_type",
        "must be synthetic or consented_human",
    ) in messages
