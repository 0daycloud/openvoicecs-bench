"""Tests for hosted-provider OpenVoiceCS adapters."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.benchmark.openvoicecs import OpenVoiceCSBench
from src.evaluation.benchmark.provider_adapters import (
    ProviderSpec,
    build_trace_prompt,
    estimate_cost_usd,
    parse_provider_response_text,
    provider_metadata,
)
from src.evaluation.benchmark.submission import score_provider


def test_build_trace_prompt_includes_tools_and_customer_text():
    scenario = OpenVoiceCSBench.load().scenarios[0]

    system, user = build_trace_prompt(scenario, trial_index=2)

    assert "Produce only valid JSON" in system
    assert scenario["conversation"][0]["text"] in user
    assert "verify_identity" in user
    assert "state_effects" in user
    assert "accounts.acct_1001.identity_verified" in user
    assert '"trial_index": 2' in user


def test_parse_provider_response_text_handles_markdown_json():
    payload = {
        "messages": [{"role": "agent", "text": "I issued the refund."}],
        "tool_calls": [{"name": "issue_refund", "arguments": {"order_id": "ord_7001"}}],
        "events": ["identity_verified"],
        "claims": [{"text": "Refund issued.", "supported": True}],
    }

    trace = parse_provider_response_text(f"```json\n{json.dumps(payload)}\n```")

    assert trace["messages"][0]["text"] == "I issued the refund."
    assert trace["tool_calls"][0]["name"] == "issue_refund"
    assert trace["events"] == ["identity_verified"]
    assert trace["claims"][0]["supported"] is True


def test_parse_provider_response_text_accepts_legacy_experience_overall():
    payload = {
        "response": "Done.",
        "experience_judgment": {"overall": 1, "notes": "legacy field"},
    }

    trace = parse_provider_response_text(json.dumps(payload))

    assert trace["messages"][0]["text"] == "Done."
    assert trace["experience_judgment"]["score"] == 1


def test_estimate_cost_usd_from_usage_and_pricing():
    cost = estimate_cost_usd(
        {"input_tokens": 1000, "output_tokens": 250},
        {"input_per_mtok": 2.0, "output_per_mtok": 8.0},
    )

    assert cost == 0.004


def test_provider_metadata_records_adapter_identity():
    spec = ProviderSpec(
        provider="openai",
        model_id="gpt-test",
        display_name="GPT Test",
        reasoning_effort="high",
    )

    metadata = provider_metadata(
        spec,
        input_modality="text",
        pricing_profile_id="openai:gpt-test",
        pricing_snapshot_date="2026-06-12",
    )

    assert metadata["provider"] == "openai"
    assert metadata["model_id"] == "gpt-test"
    assert metadata["display_name"] == "GPT Test"
    assert metadata["adapter"] == "openvoicecs-provider-adapter-v0.1"
    assert metadata["reasoning_effort"] == "high"
    assert metadata["pricing_profile_id"] == "openai:gpt-test"


def test_score_provider_with_monkeypatched_agent(monkeypatch, tmp_path: Path):
    scenario = OpenVoiceCSBench.load().scenarios[0]
    suite = {
        "name": "OpenVoiceCS-Bench",
        "version": "0.1.0",
        "scenarios": [scenario],
    }
    scenario_path = tmp_path / "scenarios.json"
    scenario_path.write_text(json.dumps(suite), encoding="utf-8")

    def fake_agent(_spec):
        def run(scenario: dict, trial_index: int) -> dict:
            del trial_index
            oracle = scenario["oracle"]
            events = []
            for event in oracle.get("required_events", []):
                if event not in events:
                    events.append(event)
            for section in ("privacy", "auth"):
                for event in oracle.get(section, {}).get("required_events", []):
                    if event not in events:
                        events.append(event)
            return {
                "messages": [{"role": "agent", "text": oracle["reference_response"]}],
                "tool_calls": oracle["expected_tool_calls"],
                "events": events,
                "usage": {"input_tokens": 1000, "output_tokens": 250},
                "cost_usd": 0.004,
            }

        return run

    monkeypatch.setattr(
        "src.evaluation.benchmark.submission.build_provider_agent",
        fake_agent,
    )
    spec = ProviderSpec(
        provider="openai",
        model_id="gpt-test",
        pricing={"input_per_mtok": 2.0, "output_per_mtok": 8.0},
    )

    report = score_provider(spec, scenario_path=scenario_path, max_items=1, trials=1)

    assert report["overall_score"] == 100.0
    assert report["model_metadata"]["provider"] == "openai"
    assert report["model_metadata"]["adapter"] == "openvoicecs-provider-adapter-v0.1"
    assert report["operational_metrics"]["avg_cost_usd"] == 0.004
