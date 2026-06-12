"""Tests for pinned pricing manifests."""

from __future__ import annotations

from src.evaluation.benchmark.frontier import build_frontier_report
from src.evaluation.benchmark.pricing import (
    load_pricing_manifest,
    pricing_manifest_stats,
    resolve_report_pricing,
    validate_pricing_manifest,
    validate_pricing_manifest_file,
)


def test_seed_pricing_manifest_validates_and_loads():
    manifest = load_pricing_manifest()

    assert validate_pricing_manifest_file() == []
    assert manifest["snapshot_date"] == "2026-06-11"
    assert manifest["currency"] == "USD"
    assert len(manifest["entries"]) == 12
    assert manifest["profiles"][0]["id"] == "reference-zero-v0.1"
    assert manifest["profiles"][1]["pipeline_type"] == "native_speech_to_speech"
    stats = pricing_manifest_stats(manifest)
    assert stats["num_profiles"] == 7
    assert stats["num_comparable_profiles"] == 5


def test_pricing_stats_count_only_non_reference_comparable_profiles():
    manifest = {
        "snapshot_date": "2026-06-11",
        "currency": "USD",
        "entries": [],
        "profiles": [
            {"id": "reference-zero", "provider": "reference", "model_id": "zero"},
            {"id": "missing-model", "provider": "provider-a"},
            {"id": "provider-a-cascade", "provider": "provider-a", "model_id": "agent-v1"},
        ],
    }

    stats = pricing_manifest_stats(manifest)

    assert stats["num_profiles"] == 3
    assert stats["num_comparable_profiles"] == 1


def test_validate_pricing_manifest_reports_missing_components():
    issues = validate_pricing_manifest({
        "snapshot_date": "2026-06-11",
        "currency": "USD",
        "entries": [
            {
                "id": "only-llm",
                "provider": "x",
                "model_id": "y",
                "component": "llm",
                "pricing": {"input_per_mtok": 1.0},
            }
        ],
        "profiles": [
            {
                "id": "bad-profile",
                "components": {"llm": "only-llm"},
            }
        ],
    })

    messages = {(issue.path, issue.message) for issue in issues}

    assert ("profiles[0].components", "missing components: asr, telephony, transport, tts") in messages
    assert ("entries", "missing component entries: asr, telephony, transport, tts") in messages


def test_validate_pricing_manifest_rejects_component_rate_key_mismatches():
    issues = validate_pricing_manifest({
        "snapshot_date": "2026-06-11",
        "currency": "USD",
        "entries": [
            {
                "id": "asr-wrong",
                "provider": "p",
                "model_id": "asr",
                "component": "asr",
                "pricing": {"input_per_mtok": 1.0},
            },
            {
                "id": "llm-wrong",
                "provider": "p",
                "model_id": "llm",
                "component": "llm",
                "pricing": {"tts_per_minute": 0.01},
            },
            {
                "id": "tts-wrong",
                "provider": "p",
                "model_id": "tts",
                "component": "tts",
                "pricing": {"telephony_per_minute": 0.01},
            },
            {
                "id": "tel-wrong",
                "provider": "p",
                "model_id": "tel",
                "component": "telephony",
                "pricing": {"transport_per_minute": 0.01},
            },
            {
                "id": "net-wrong",
                "provider": "p",
                "model_id": "net",
                "component": "transport",
                "pricing": {"asr_per_minute": 0.01},
            },
        ],
        "profiles": [
            {
                "id": "paid",
                "components": {
                    "asr": "asr-wrong",
                    "llm": "llm-wrong",
                    "tts": "tts-wrong",
                    "telephony": "tel-wrong",
                    "transport": "net-wrong",
                },
            }
        ],
    })

    messages = {(issue.scenario_id, issue.path, issue.message) for issue in issues}

    assert ("asr-wrong", "entries[0].pricing", "must include a numeric asr pricing key") in messages
    assert ("llm-wrong", "entries[1].pricing", "must include a numeric llm pricing key") in messages
    assert ("tts-wrong", "entries[2].pricing", "must include a numeric tts pricing key") in messages
    assert (
        "tel-wrong",
        "entries[3].pricing",
        "must include a numeric telephony pricing key",
    ) in messages
    assert (
        "net-wrong",
        "entries[4].pricing",
        "must include a numeric transport pricing key",
    ) in messages


def test_validate_pricing_manifest_rejects_profile_component_entry_mismatches():
    issues = validate_pricing_manifest({
        "snapshot_date": "2026-06-11",
        "currency": "USD",
        "entries": [
            {
                "id": "asr",
                "provider": "p",
                "model_id": "asr",
                "component": "asr",
                "pricing": {"asr_per_minute": 0.01},
            },
            {
                "id": "llm",
                "provider": "p",
                "model_id": "llm",
                "component": "llm",
                "pricing": {"input_per_mtok": 1.0},
            },
            {
                "id": "tts",
                "provider": "p",
                "model_id": "tts",
                "component": "tts",
                "pricing": {"tts_per_minute": 0.01},
            },
            {
                "id": "tel",
                "provider": "p",
                "model_id": "tel",
                "component": "telephony",
                "pricing": {"telephony_per_minute": 0.01},
            },
            {
                "id": "net",
                "provider": "p",
                "model_id": "net",
                "component": "transport",
                "pricing": {"transport_per_minute": 0.01},
            },
        ],
        "profiles": [
            {
                "id": "bad-profile",
                "components": {
                    "asr": "llm",
                    "llm": "asr",
                    "tts": "tts",
                    "telephony": "tel",
                    "transport": "net",
                },
            }
        ],
    })

    messages = {(issue.scenario_id, issue.path, issue.message) for issue in issues}

    assert (
        "bad-profile",
        "profiles[0].components.asr",
        "entry component mismatch: expected asr, got llm",
    ) in messages
    assert (
        "bad-profile",
        "profiles[0].components.llm",
        "entry component mismatch: expected llm, got asr",
    ) in messages


def test_resolve_report_pricing_from_profile_merges_component_rates():
    manifest = load_pricing_manifest()
    pricing = resolve_report_pricing(
        {"model_metadata": {"pricing_profile_id": "reference-zero-v0.1"}},
        manifest,
    )

    assert pricing["snapshot_date"] == "2026-06-11"
    assert pricing["profile_id"] == "reference-zero-v0.1"
    assert pricing["asr_per_minute"] == 0.0
    assert pricing["input_per_mtok"] == 0.0
    assert pricing["tts_per_1k_characters"] == 0.0
    assert pricing["telephony_per_minute"] == 0.0
    assert pricing["transport_per_minute"] == 0.0


def test_frontier_uses_pricing_manifest_when_report_has_usage_without_pricing():
    manifest = {
        "snapshot_date": "2026-06-11",
        "currency": "USD",
        "entries": [
            {"id": "asr", "provider": "p", "model_id": "asr", "component": "asr", "pricing": {"asr_per_minute": 0.01}},
            {"id": "llm", "provider": "p", "model_id": "llm", "component": "llm", "pricing": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}},
            {"id": "tts", "provider": "p", "model_id": "tts", "component": "tts", "pricing": {"tts_per_1k_characters": 0.01}},
            {"id": "tel", "provider": "p", "model_id": "tel", "component": "telephony", "pricing": {"telephony_per_minute": 0.02}},
            {"id": "net", "provider": "p", "model_id": "net", "component": "transport", "pricing": {"transport_per_minute": 0.03}},
        ],
        "profiles": [
            {
                "id": "paid",
                "components": {
                    "asr": "asr",
                    "llm": "llm",
                    "tts": "tts",
                    "telephony": "tel",
                    "transport": "net",
                },
            }
        ],
    }
    report = {
        "model_metadata": {"display_name": "manifest-priced", "pricing_profile_id": "paid"},
        "metric_scores": {"task_success": 1.0, "experience_proxy": 1.0},
        "results": [
            {
                "id": "s1",
                "domain": "retail",
                "trials": [
                    {
                        "latency_ms": 100,
                        "usage": {
                            "asr_seconds": 60,
                            "input_tokens": 1000,
                            "output_tokens": 500,
                            "tts_characters": 1000,
                            "call_duration_seconds": 60,
                            "transport_seconds": 60,
                        },
                    }
                ],
            }
        ],
    }

    frontier = build_frontier_report([report], pricing_manifest=manifest)
    scorecard = frontier["scorecards"]["manifest-priced"]

    assert frontier["pricing_snapshot"]["snapshot_date"] == "2026-06-11"
    assert frontier["systems"]["manifest-priced"]["resolved_pricing"]["profile_id"] == "paid"
    assert scorecard["cost_usd_per_successful_conversation"] == 0.072
    assert scorecard["cost_provenance"]["pricing_source"] == "profile"
    assert scorecard["cost_provenance"]["pipeline_type"] == "cascaded"
    assert scorecard["cost_provenance"]["required_components"] == [
        "asr",
        "llm",
        "tts",
        "telephony",
        "transport",
    ]
    assert scorecard["cost_provenance"]["fully_loaded_samples"] == 1


def test_native_speech_to_speech_pricing_profile_derives_cost():
    manifest = {
        "snapshot_date": "2026-06-11",
        "currency": "USD",
        "entries": [
            {
                "id": "s2s",
                "provider": "p",
                "model_id": "native",
                "component": "speech_to_speech",
                "pricing": {
                    "speech_to_speech_per_minute": 0.10,
                    "input_audio_per_minute": 0.06,
                    "output_audio_per_minute": 0.12,
                    "input_audio_per_mtok": 10.0,
                    "output_audio_per_mtok": 20.0,
                },
            },
            {
                "id": "tel",
                "provider": "p",
                "model_id": "tel",
                "component": "telephony",
                "pricing": {"telephony_per_minute": 0.02},
            },
            {
                "id": "net",
                "provider": "p",
                "model_id": "net",
                "component": "transport",
                "pricing": {"transport_per_minute": 0.03},
            },
        ],
        "profiles": [
            {
                "id": "native-paid",
                "pipeline_type": "native_speech_to_speech",
                "components": {
                    "speech_to_speech": "s2s",
                    "telephony": "tel",
                    "transport": "net",
                },
            }
        ],
    }
    report = {
        "model_metadata": {"display_name": "native", "pricing_profile_id": "native-paid"},
        "metric_scores": {"task_success": 1.0, "experience_proxy": 1.0},
        "results": [
            {
                "id": "s1",
                "domain": "retail",
                "trials": [
                    {
                        "latency_ms": 100,
                        "usage": {
                            "speech_to_speech_seconds": 60,
                            "input_audio_seconds": 30,
                            "output_audio_seconds": 20,
                            "input_audio_tokens": 1000,
                            "output_audio_tokens": 500,
                            "call_duration_seconds": 60,
                            "transport_seconds": 60,
                        },
                    }
                ],
            }
        ],
    }

    assert validate_pricing_manifest(manifest) == []

    frontier = build_frontier_report([report], pricing_manifest=manifest)
    system = frontier["systems"]["native"]
    scorecard = frontier["scorecards"]["native"]

    assert system["resolved_pricing"]["pipeline_type"] == "native_speech_to_speech"
    assert scorecard["cost_provenance"]["pipeline_type"] == "native_speech_to_speech"
    assert scorecard["cost_provenance"]["required_components"] == [
        "speech_to_speech",
        "telephony",
        "transport",
    ]
    assert scorecard["cost_provenance"]["fully_loaded_samples"] == 1
    assert scorecard["avg_component_cost_usd"]["speech_to_speech"] == 0.19
    assert scorecard["cost_usd_per_successful_conversation"] == 0.24
