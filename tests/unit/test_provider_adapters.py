"""Tests for hosted-provider OpenVoiceCS adapters."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from src.evaluation.benchmark.openvoicecs import OpenVoiceCSBench
from src.evaluation.benchmark.provider_adapters import (
    ProviderSpec,
    _derive_events,
    _execute_scenario_tool,
    _openai_tool_schemas,
    _parse_native_final_trace,
    build_json_action_prompt,
    build_provider_agent,
    build_provider_spec,
    build_trace_prompt,
    estimate_cost_usd,
    parse_json_action_response,
    parse_provider_response_text,
    provider_metadata,
)
from src.evaluation.benchmark.submission import score_provider


def test_build_trace_prompt_includes_tools_and_customer_text():
    scenario = OpenVoiceCSBench.load().scenarios[0]

    system, user = build_trace_prompt(scenario, trial_index=2)

    assert "evaluated" not in system.lower()
    assert "benchmark" not in system.lower()
    assert scenario["conversation"][0]["text"] in user
    assert "verify_identity" in user
    assert "required_events" not in user
    assert "forbidden_events" not in user
    assert "forbidden_tool_calls" not in user
    assert "state_effects" not in user
    assert "preconditions" not in user
    assert "state_updates" not in user
    assert '"identity_verified": false' in user
    assert '"trial_index": 2' in user
    scenario_view = json.loads(user.split("Customer session:\n", 1)[1])
    assert scenario_view["available_tools"][0]["parameters"] == {"account_id": "string"}
    assert "required_arguments" not in scenario_view["available_tools"][0]


def test_native_trace_prompt_does_not_ask_for_tool_or_event_labels():
    scenario = OpenVoiceCSBench.load().scenarios[0]

    _system, user = build_trace_prompt(scenario, trial_index=0, native_tools=True)
    response_contract = user.split("Customer session:", 1)[0]

    assert "tool_calls" not in response_contract
    assert "events" not in response_contract
    assert "claims" not in response_contract
    assert "experience_judgment" not in response_contract


def test_openai_tool_schemas_are_generic_typed_without_constants():
    scenario = OpenVoiceCSBench.load().scenarios[0]

    schemas = _openai_tool_schemas(scenario)
    encoded = json.dumps(schemas)

    assert "const" not in encoded
    assert schemas[0]["function"]["parameters"]["properties"]["account_id"] == {"type": "string"}
    assert "state_updates" not in encoded
    assert "preconditions" not in encoded
    create_case_schema = next(item for item in schemas if item["function"]["name"] == "create_case")
    create_case_params = create_case_schema["function"]["parameters"]
    # System-assigned arguments stay visible so the agent can tell what the tool
    # records, but are excluded from `required` so it is never asked to invent a
    # value it has no way to know.
    generated = {
        name
        for name, schema in create_case_params["properties"].items()
        if "Assigned by the system" in (schema.get("description") or "")
    }
    assert generated == {"case_id"}, "generated arguments must still be advertised to the model"
    # `reason` is a documented classification vocabulary (argument_enums), not
    # an opaque system ID, so unlike `case_id` it is a real, required,
    # enum-constrained argument.
    assert create_case_params["required"] == ["account_id", "reason"]
    assert generated.isdisjoint(create_case_params["required"])
    assert set(create_case_params["properties"]) == {"account_id", "reason"} | generated
    assert "damaged_item" in create_case_params["properties"]["reason"]["enum"]


def test_openai_tool_schemas_emit_enum_and_require_documented_classification_fields():
    """A field with a declared vocabulary is a real, answerable decision.

    Unlike ``generated_arguments`` (system-assigned, excused from ``required``,
    labeled "assigned by the system"), an ``argument_enums`` field is something
    the model can and must choose correctly, so it belongs in ``required`` with
    its valid choices spelled out — not hidden as optional.
    """
    scenario = {
        "tools": [
            {
                "name": "issue_refund",
                "required_arguments": {"order_id": "ord_1", "reason": "damaged_item"},
                "generated_arguments": {},
                "argument_enums": {"reason": ["damaged_gift_item", "damaged_item"]},
            }
        ]
    }

    schemas = _openai_tool_schemas(scenario)
    params = schemas[0]["function"]["parameters"]

    assert params["properties"]["reason"] == {
        "type": "string",
        "enum": ["damaged_gift_item", "damaged_item"],
    }
    assert "reason" in params["required"]
    assert "order_id" in params["required"]


def test_openai_tool_schemas_enum_field_is_never_treated_as_system_assigned():
    """Guard the interaction between the two annotations on one tool."""
    scenario = {
        "tools": [
            {
                "name": "create_case",
                "required_arguments": {
                    "case_id": "case_1",
                    "account_id": "acct_1",
                    "reason": "card_dispute",
                },
                "generated_arguments": {"case_id": "case_1"},
                "argument_enums": {"reason": ["card_dispute", "merchant_hold"]},
            }
        ]
    }

    schemas = _openai_tool_schemas(scenario)
    params = schemas[0]["function"]["parameters"]

    assert set(params["required"]) == {"account_id", "reason"}
    assert params["properties"]["reason"]["enum"] == ["card_dispute", "merchant_hold"]
    assert "description" not in params["properties"]["reason"]
    assert params["properties"]["case_id"]["description"] == "Assigned by the system; may be omitted."
    assert "enum" not in params["properties"]["case_id"]


def test_json_action_prompt_surfaces_enum_choices_for_classification_arguments():
    """The stepwise JSON-action path has no native ``tools`` schema, so the
    closed vocabulary has to be spelled out in the argument type text instead
    — otherwise chat-only models (the long tail this loop exists for) stay as
    blind to the vocabulary as they were before this fix.
    """
    scenario = {
        "id": "unit-test-enum-scenario",
        "domain": "retail",
        "customer_goal": "Refund a damaged item.",
        "conversation": [{"role": "customer", "text": "My order arrived damaged."}],
        "initial_state": {},
        "policy": {},
        "tools": [
            {
                "name": "issue_refund",
                "required_arguments": {"order_id": "ord_1", "reason": "damaged_item"},
                "argument_enums": {"reason": ["damaged_gift_item", "damaged_item"]},
            }
        ],
    }

    _system, user = build_json_action_prompt(scenario, trial_index=0)

    scenario_view = json.loads(user.split("Customer session:\n", 1)[1])
    reason_type = scenario_view["available_tools"][0]["parameters"]["reason"]
    assert "damaged_gift_item" in reason_type
    assert "damaged_item" in reason_type
    order_id_type = scenario_view["available_tools"][0]["parameters"]["order_id"]
    assert order_id_type == "string"


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


def test_json_action_prompt_uses_stepwise_protocol():
    scenario = OpenVoiceCSBench.load().scenarios[0]

    _system, user = build_json_action_prompt(scenario, trial_index=1)

    response_contract = user.split("Customer session:", 1)[0]
    assert '"action":"call_tool"' in response_contract
    assert '"action":"final"' in response_contract
    assert "tool result" in response_contract.lower()
    assert "Do not invent tool results" in response_contract
    assert '"trial_index": 1' in user


def test_parse_json_action_response_accepts_call_tool_and_final():
    call = parse_json_action_response(
        '```json\n{"action":"call_tool","name":"verify_identity","arguments":{"account_id":"acct_1"}}\n```'
    )
    final = parse_json_action_response('{"action":"final","message":"Done."}')

    assert call == {
        "action": "call_tool",
        "name": "verify_identity",
        "arguments": {"account_id": "acct_1"},
    }
    assert final == {"action": "final", "message": "Done."}


def test_parse_json_action_response_tolerates_legacy_trace_shape():
    action = parse_json_action_response(
        json.dumps({
            "messages": [{"role": "agent", "text": "I can help."}],
            "tool_calls": [{"name": "verify_identity", "arguments": {"account_id": "acct_1"}}],
        })
    )

    assert action == {
        "action": "call_tool",
        "name": "verify_identity",
        "arguments": {"account_id": "acct_1"},
    }


def test_parse_json_action_response_treats_plain_text_as_final():
    action = parse_json_action_response("I can help with that.")

    assert action == {"action": "final", "message": "I can help with that."}


def test_parse_json_action_response_accepts_common_aliases():
    tool_action = parse_json_action_response(
        json.dumps({
            "action": "tool_call",
            "tool_name": "verify_identity",
            "args": {"account_id": "acct_1"},
        })
    )
    final_action = parse_json_action_response(json.dumps({"action": "respond", "text": "Done."}))

    assert tool_action == {
        "action": "call_tool",
        "name": "verify_identity",
        "arguments": {"account_id": "acct_1"},
    }
    assert final_action == {"action": "final", "message": "Done."}


def test_parse_json_action_response_accepts_tool_name_as_action():
    action = parse_json_action_response(
        json.dumps({"action": "verify_identity", "arguments": {"account_id": "acct_1"}})
    )

    assert action == {
        "action": "call_tool",
        "name": "verify_identity",
        "arguments": {"account_id": "acct_1"},
    }


def test_native_final_trace_accepts_empty_or_plain_text():
    empty = _parse_native_final_trace("")
    plain = _parse_native_final_trace("I completed the request.")

    assert empty == {"messages": [], "tool_calls": [], "events": []}
    assert plain["messages"] == [{"role": "agent", "text": "I completed the request."}]


def test_openai_compatible_json_action_loop_executes_tools(monkeypatch):
    requests = []
    responses = [
        {"action": "call_tool", "name": "verify_identity", "arguments": {"account_id": "acct_1"}},
        {"action": "call_tool", "name": "create_case", "arguments": {"account_id": "acct_1"}},
        {"action": "final", "message": "Your case is created."},
    ]

    class FakeCompletions:
        def create(self, **request):
            requests.append(request)
            payload = responses.pop(0)
            message = SimpleNamespace(content=json.dumps(payload))
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    scenario = {
        "id": "case-test",
        "domain": "retail",
        "customer_goal": "Create a case.",
        "conversation": [{"role": "customer", "text": "Please open a case."}],
        "initial_state": {
            "accounts": {"acct_1": {"identity_verified": False}},
            "cases": {},
        },
        "policy": {},
        "tools": [
            {
                "name": "verify_identity",
                "required_arguments": {"account_id": "acct_1"},
                "state_updates": [{"path": "accounts.acct_1.identity_verified", "value": True}],
            },
            {
                "name": "create_case",
                "required_arguments": {"case_id": "case_1", "account_id": "acct_1"},
                "generated_arguments": {"case_id": "case_1"},
                "preconditions": [{"path": "accounts.acct_1.identity_verified", "value": True}],
                "state_updates": [{"path": "cases.case_1.status", "value": "open"}],
            },
        ],
    }
    agent = build_provider_agent(
        ProviderSpec(
            provider="openrouter",
            model_id="fake/model",
            api_key="test-key",
            native_tools=False,
        )
    )

    trace = agent(scenario, 0)

    assert trace["tool_calls"] == [
        {"name": "verify_identity", "arguments": {"account_id": "acct_1"}},
        {"name": "create_case", "arguments": {"account_id": "acct_1"}},
    ]
    assert trace["tool_results"][0]["ok"] is True
    assert trace["tool_results"][1]["generated_arguments"] == {"case_id": "case_1"}
    assert trace["messages"] == [{"role": "agent", "text": "Your case is created."}]
    assert trace["usage"] == {"input_tokens": 30, "output_tokens": 15}
    assert any("tool_result" in message["content"] for message in requests[1]["messages"])


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
    assert metadata["native_tools"] is False
    assert metadata["pricing_profile_id"] == "openai:gpt-test"


def test_build_provider_spec_records_native_tool_mode():
    spec = build_provider_spec(
        "openai",
        model_id="gpt-test",
        reasoning_effort="low",
    )

    metadata = provider_metadata(spec, input_modality="text")

    assert metadata["reasoning_effort"] == "low"
    assert metadata["native_tools"] is True


def test_build_provider_spec_keeps_json_trace_fallback():
    spec = build_provider_spec(
        "openrouter",
        model_id="openai/gpt-test",
        native_tools=False,
    )

    metadata = provider_metadata(spec, input_modality="text")

    assert metadata["native_tools"] is False


def test_build_provider_spec_does_not_default_native_for_non_openai_compatible_provider():
    spec = build_provider_spec("google", model_id="gemini-test")

    metadata = provider_metadata(spec, input_modality="text")

    assert metadata["native_tools"] is False


def test_openrouter_native_tools_uses_native_tool_agent(monkeypatch):
    calls = []

    def fake_native_agent(spec):
        calls.append(("native", spec.provider, spec.model_id))
        return lambda _scenario, _trial_index: {}

    def fake_json_agent(_spec):
        raise AssertionError("native tool mode should not use JSON trace adapter")

    monkeypatch.setattr(
        "src.evaluation.benchmark.provider_adapters._build_openai_native_tool_agent",
        fake_native_agent,
    )
    monkeypatch.setattr(
        "src.evaluation.benchmark.provider_adapters._build_openai_compatible_agent",
        fake_json_agent,
    )

    agent = build_provider_agent(
        ProviderSpec(
            provider="openrouter",
            model_id="google/gemini-3-flash-preview",
            native_tools=True,
        )
    )

    assert callable(agent)
    assert calls == [("native", "openrouter", "google/gemini-3-flash-preview")]


def test_native_tool_execution_models_external_failure():
    scenario = {
        "initial_state": {
            "orders": {"ord_1": {"refund_status": "none"}},
            "external_systems": {"refund_processor": {"last_error": None}},
        },
        "tools": [
            {
                "name": "issue_refund",
                "required_arguments": {"order_id": "ord_1"},
                "state_updates": [{"path": "orders.ord_1.refund_status", "value": "issued"}],
                "failure": {
                    "type": "external_unavailable",
                    "code": "REFUND_PROCESSOR_503",
                    "message": "refund processor unavailable",
                    "retryable": True,
                    "state_updates": [
                        {
                            "path": "external_systems.refund_processor.last_error",
                            "value": "REFUND_PROCESSOR_503",
                        }
                    ],
                },
            }
        ],
    }
    state = json.loads(json.dumps(scenario["initial_state"]))

    result = _execute_scenario_tool(
        scenario,
        state,
        "issue_refund",
        {"order_id": "ord_1"},
    )

    assert result["ok"] is False
    assert result["error"] == "external_unavailable"
    assert "state" not in result
    assert "code" not in result
    assert state["orders"]["ord_1"]["refund_status"] == "none"
    assert state["external_systems"]["refund_processor"]["last_error"] == "REFUND_PROCESSOR_503"


def test_native_tool_execution_accepts_server_generated_arguments():
    scenario = {
        "initial_state": {"cases": {}},
        "tools": [
            {
                "name": "create_case",
                "required_arguments": {
                    "case_id": "case_1",
                    "account_id": "acct_1",
                    "reason": "damaged_item",
                },
                "generated_arguments": {
                    "case_id": "case_1",
                    "reason": "damaged_item",
                },
                "state_updates": [{"path": "cases.case_1.status", "value": "created"}],
                "result": {"case_id": "case_1", "status": "created"},
            }
        ],
    }
    state = json.loads(json.dumps(scenario["initial_state"]))

    result = _execute_scenario_tool(scenario, state, "create_case", {"account_id": "acct_1"})

    assert result["ok"] is True
    assert result["generated_arguments"] == {"case_id": "case_1", "reason": "damaged_item"}
    assert state["cases"]["case_1"]["status"] == "created"


def test_native_tool_execution_requires_bound_id_from_prior_result():
    scenario = {
        "initial_state": {"cases": {}, "escalations": {}},
        "tools": [
            {
                "name": "create_case",
                "required_arguments": {"case_id": "case_1", "account_id": "acct_1"},
                "generated_arguments": {"case_id": "case_1"},
                "state_updates": [{"path": "cases.case_1.status", "value": "created"}],
                "result": {"case_id": "case_1", "status": "created"},
            },
            {
                "name": "escalate_to_human",
                "required_arguments": {
                    "escalation_id": "esc_1",
                    "account_id": "acct_1",
                    "case_id": "case_1",
                },
                "generated_arguments": {"escalation_id": "esc_1"},
                "argument_bindings": {
                    "case_id": {"tool": "create_case", "path": "result.case_id"}
                },
                "state_updates": [{"path": "escalations.esc_1.status", "value": "assigned"}],
                "result": {"escalation_id": "esc_1", "status": "assigned"},
            },
        ],
    }
    state = json.loads(json.dumps(scenario["initial_state"]))

    first = _execute_scenario_tool(scenario, state, "create_case", {"account_id": "acct_1"})
    second = _execute_scenario_tool(
        scenario,
        state,
        "escalate_to_human",
        {"account_id": "acct_1", "case_id": "case_1"},
        tool_results=[first],
    )
    missing_bound = _execute_scenario_tool(
        scenario,
        state,
        "escalate_to_human",
        {"account_id": "acct_1"},
        tool_results=[first],
    )

    assert second["ok"] is True
    assert state["escalations"]["esc_1"]["status"] == "assigned"
    assert missing_bound["error"] == "argument_binding_mismatch"
    assert missing_bound["binding_errors"][0]["argument"] == "case_id"


def test_derive_events_from_tools_and_final_response():
    scenario = OpenVoiceCSBench.load().scenarios[0]
    tool_calls = [
        {"name": "verify_identity", "arguments": {"account_id": "acct_1001"}},
        {
            "name": "issue_refund",
            "arguments": {
                "order_id": "ord_7001",
                "amount_cents": 5299,
                "reason": "damaged_item",
            },
        },
        {
            "name": "create_case",
            "arguments": {
                "case_id": "case_9001",
                "account_id": "acct_1001",
                "reason": "damaged_item",
            },
        },
    ]
    messages = [{"role": "agent", "text": "I verified you and processed the damaged item refund."}]

    events = _derive_events(scenario, tool_calls, messages)

    assert "identity_verified" in events
    assert "damage_attested" in events
    assert "pii_minimization" in events


def test_derive_events_covers_travel_fee_waiver_language():
    scenario = next(
        item
        for item in OpenVoiceCSBench.load().scenarios
        if item["id"] == "travel-rebook-missed-connection-001"
    )
    tool_calls = [
        {"name": "verify_identity", "arguments": {"account_id": "acct_2001"}},
        {
            "name": "rebook_flight",
            "arguments": {
                "booking_id": "bk_3001",
                "new_flight_id": "fl_881",
                "fee_cents": 0,
            },
        },
    ]
    messages = [
        {
            "role": "agent",
            "text": "Since this was caused by the airline delay, there is no fee for the rebooking.",
        }
    ]

    events = _derive_events(scenario, tool_calls, messages)

    assert "airline_delay_confirmed" in events
    assert "fee_waiver_applied" in events


def test_derive_events_covers_saas_admin_and_security_hold_language():
    scenario = next(
        item
        for item in OpenVoiceCSBench.load().scenarios
        if item["id"] == "saas-account-access-001"
    )
    tool_calls = [
        {"name": "verify_identity", "arguments": {"account_id": "acct_7001"}},
        {
            "name": "start_mfa_reset",
            "arguments": {"account_id": "acct_7001", "workspace_id": "ws_8101"},
        },
    ]
    messages = [
        {
            "role": "agent",
            "text": "I verified your workspace admin role and started the MFA reset. A security hold is active.",
        }
    ]

    events = _derive_events(scenario, tool_calls, messages)

    assert "admin_role_confirmed" in events
    assert "security_hold_explained" in events


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
