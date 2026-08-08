"""Tests for the hybrid semantic grounding fallback in check_factual_grounding.

check_factual_grounding runs the original literal/regex matcher first
(unchanged). Required claims it cannot confirm, and forbidden claims that
share real vocabulary with a pattern without matching it exactly, fall back to
one batched semantic-judge call per trace. This is the default (hybrid) mode;
mode="legacy" (or OPENVOICECS_GROUNDING_MODE=legacy) disables the fallback and
reproduces the pre-semantic phrase-matcher scores documented in
docs/known-limitations.md section 7.

The semantic judge is always a caller stub here -- these tests must not
depend on network access or a provider API key.
"""

from __future__ import annotations

import json

import pytest

from src.evaluation.benchmark.openvoicecs import (
    DEFAULT_GROUNDING_JUDGE_MODEL_ID,
    DEFAULT_GROUNDING_JUDGE_PROVIDER,
    OpenVoiceCSBench,
    _default_grounding_judge_spec,
    _forbidden_claim_near_miss,
    check_factual_grounding,
    classify_trial_error,
    oracle_agent,
)


def _grounding_scenario(*, required_claims=None, forbidden_claims=None, max_hallucinations_per_turn=0):
    return {
        "customer_goal": "Resolve the customer's issue.",
        "oracle": {
            "grounding": {
                "required_claims": required_claims if required_claims is not None else [
                    {"id": "fee_waived", "any_terms": ["no change fee", "no fee", "fee waiver"]},
                ],
                "forbidden_claims": forbidden_claims if forbidden_claims is not None else [
                    {
                        "id": "cash_compensation_promised",
                        "patterns": ["cash compensation", "refund for the flight"],
                        "severity": "medium",
                    },
                ],
                "max_hallucinations_per_turn": max_hallucinations_per_turn,
            },
        },
    }


def _trace(text: str) -> dict:
    return {"messages": [{"role": "agent", "text": text}], "claims": []}


def _stub_caller(response_payload: dict):
    def caller(spec, messages, max_output_tokens, temperature, timeout_seconds):
        del spec, messages, max_output_tokens, temperature, timeout_seconds
        return json.dumps(response_payload)
    return caller


# ---------------------------------------------------------------------------
# Oracle offline guarantee: score --agent oracle must stay usable without a
# judge API key. Covered as two separate assertions, one per fallback path,
# because the two are gated by different conditions (missing_required is
# empty by construction for the oracle; near_miss_forbidden requires no
# regex-unmatched pattern to share every keyword with the oracle's text) and
# either could regress independently of the other.
# ---------------------------------------------------------------------------


def test_oracle_agent_never_triggers_required_claim_semantic_fallback():
    bench = OpenVoiceCSBench.load()
    calls = []

    def caller(spec, messages, max_output_tokens, temperature, timeout_seconds):
        del spec, max_output_tokens, temperature, timeout_seconds
        payload = json.loads(messages[1]["content"])
        calls.append(payload)
        return json.dumps({
            "required_claims": [{"id": c["id"], "grounded": True, "reason": "stub"} for c in payload["required_claims"]],
            "forbidden_claims": [{"id": c["id"], "violated": False, "reason": "stub"} for c in payload["forbidden_claims"]],
        })

    scenarios_with_required_claims = 0
    for scenario in bench.scenarios:
        grounding = (scenario.get("oracle") or {}).get("grounding", {})
        if grounding.get("required_claims"):
            scenarios_with_required_claims += 1
        trace = oracle_agent(scenario)
        result = check_factual_grounding(trace, scenario, caller=caller)
        assert result["missing_required_claims"] == [], scenario["id"]

    assert scenarios_with_required_claims > 0
    required_side_calls = [payload for payload in calls if payload["required_claims"]]
    assert required_side_calls == []


def test_oracle_agent_never_triggers_forbidden_claim_semantic_fallback():
    bench = OpenVoiceCSBench.load()
    calls = []

    def caller(spec, messages, max_output_tokens, temperature, timeout_seconds):
        del spec, max_output_tokens, temperature, timeout_seconds
        payload = json.loads(messages[1]["content"])
        calls.append(payload)
        return json.dumps({
            "required_claims": [{"id": c["id"], "grounded": True, "reason": "stub"} for c in payload["required_claims"]],
            "forbidden_claims": [{"id": c["id"], "violated": False, "reason": "stub"} for c in payload["forbidden_claims"]],
        })

    scenarios_with_forbidden_claims = 0
    for scenario in bench.scenarios:
        grounding = (scenario.get("oracle") or {}).get("grounding", {})
        if grounding.get("forbidden_claims"):
            scenarios_with_forbidden_claims += 1
        trace = oracle_agent(scenario)
        check_factual_grounding(trace, scenario, caller=caller)

    assert scenarios_with_forbidden_claims > 0
    forbidden_side_calls = [payload for payload in calls if payload["forbidden_claims"]]
    assert forbidden_side_calls == []


# ---------------------------------------------------------------------------
# The near-miss pre-filter for forbidden claims: it must require every content
# word of a pattern, with word boundaries, or two known false-positive classes
# come back -- vocabulary genuinely shared with a required claim ("refund"),
# and simple negation ("changed" inside "unchanged").
# ---------------------------------------------------------------------------


def test_forbidden_claim_near_miss_ignores_negated_wording():
    assert _forbidden_claim_near_miss(
        "I applied the credit and left your plan unchanged.",
        ["changed your plan", "upgraded your plan"],
    ) is False


def test_forbidden_claim_near_miss_ignores_vocabulary_shared_with_required_claims():
    assert _forbidden_claim_near_miss(
        "I verified the order and processed the refund for the damaged item.",
        ["instant(ly)? (bank )?refund", "already in your bank"],
    ) is False


def test_forbidden_claim_near_miss_keeps_short_digit_tokens():
    # "14 Pine" only differs from a corrected "40 Pine" by the number,  which
    # is shorter than the general keyword length floor -- digits must never
    # be dropped by that filter or the near-miss check goes blind to exactly
    # the pre-correction-value class of forbidden claim it exists to catch.
    assert _forbidden_claim_near_miss("scheduled the technician for 40 Pine Street", ["14 Pine"]) is False
    assert _forbidden_claim_near_miss("scheduled the technician for 14 Pine Street", ["14 Pine"]) is True


def test_forbidden_claim_near_miss_requires_every_keyword_present():
    assert _forbidden_claim_near_miss(
        "I'll get you some cash back as compensation for the delay.",
        ["cash compensation"],
    ) is True
    assert _forbidden_claim_near_miss(
        "I'll get you some cash back for the delay.",
        ["cash compensation"],
    ) is False


# ---------------------------------------------------------------------------
# The three failure modes documented in docs/known-limitations.md section 7.
# ---------------------------------------------------------------------------


def test_semantic_fallback_recognizes_synonymy_the_literal_matcher_misses():
    scenario = _grounding_scenario()
    trace = _trace("I rebooked you on the next flight at no charge.")

    legacy = check_factual_grounding(trace, scenario, mode="legacy")
    assert legacy["required_passed"] is False

    caller = _stub_caller({
        "required_claims": [{"id": "fee_waived", "grounded": True, "reason": "at no charge waives the fee"}],
        "forbidden_claims": [],
    })
    hybrid = check_factual_grounding(trace, scenario, caller=caller)
    assert hybrid["required_passed"] is True
    assert hybrid["score"] == 1.0
    assert hybrid["semantic_fallback"]["required"]["fee_waived"]["grounded"] is True


def test_semantic_fallback_still_fails_a_genuine_omission():
    scenario = _grounding_scenario(
        required_claims=[{"id": "credit_amount", "any_terms": ["12 dollar", "\\$12"]}],
        forbidden_claims=[],
    )
    trace = _trace("I applied a credit to your account for the duplicate fee.")

    caller = _stub_caller({
        "required_claims": [{"id": "credit_amount", "grounded": False, "reason": "amount never stated"}],
        "forbidden_claims": [],
    })
    result = check_factual_grounding(trace, scenario, caller=caller)
    assert result["required_passed"] is False
    assert result["score"] == 0.0


def test_semantic_fallback_does_not_penalize_an_honest_failure_report():
    scenario = _grounding_scenario(
        required_claims=[{"id": "completed", "any_terms": ["completed", "processed successfully"]}],
        forbidden_claims=[],
    )
    trace = _trace("I couldn't complete this myself, so I've escalated it to a specialist.")

    caller = _stub_caller({
        "required_claims": [{"id": "completed", "grounded": False, "reason": "agent reported failure, not completion"}],
        "forbidden_claims": [],
    })
    result = check_factual_grounding(trace, scenario, caller=caller)
    # The point of this scenario type is that grounding is correctly False --
    # an honest failure report must not be laundered into a passing claim.
    assert result["required_passed"] is False


def test_semantic_fallback_catches_a_paraphrased_forbidden_claim():
    scenario = _grounding_scenario(
        required_claims=[],
        forbidden_claims=[{"id": "cash_compensation_promised", "patterns": ["cash compensation"], "severity": "medium"}],
    )
    trace = _trace("I'll get you some cash back as compensation for the delay.")

    def caller(spec, messages, max_output_tokens, temperature, timeout_seconds):
        del spec, max_output_tokens, temperature, timeout_seconds
        payload = json.loads(messages[1]["content"])
        assert payload["forbidden_claims"], "near-miss pre-filter should have escalated this claim"
        return json.dumps({
            "required_claims": [],
            "forbidden_claims": [{"id": "cash_compensation_promised", "violated": True, "reason": "promised cash as compensation"}],
        })

    result = check_factual_grounding(trace, scenario, caller=caller)
    detected_ids = [claim["id"] for claim in result["unsupported_claims_detected"]]
    assert "cash_compensation_promised" in detected_ids


# ---------------------------------------------------------------------------
# Cost: one batched call per trace, not one call per unresolved claim.
# ---------------------------------------------------------------------------


def test_semantic_fallback_batches_every_unresolved_claim_into_one_call():
    scenario = _grounding_scenario(
        required_claims=[
            {"id": "fee_waived", "any_terms": ["no fee"]},
            {"id": "credit_amount", "any_terms": ["12 dollar"]},
        ],
        forbidden_claims=[{"id": "cash_compensation_promised", "patterns": ["cash compensation"]}],
    )
    trace = _trace("I rebooked you at no extra cost and gave you cash as compensation.")

    calls = []

    def caller(spec, messages, max_output_tokens, temperature, timeout_seconds):
        del spec, max_output_tokens, temperature, timeout_seconds
        payload = json.loads(messages[1]["content"])
        calls.append(payload)
        return json.dumps({
            "required_claims": [{"id": c["id"], "grounded": False, "reason": "stub"} for c in payload["required_claims"]],
            "forbidden_claims": [{"id": c["id"], "violated": False, "reason": "stub"} for c in payload["forbidden_claims"]],
        })

    check_factual_grounding(trace, scenario, caller=caller)
    assert len(calls) == 1
    assert len(calls[0]["required_claims"]) == 2
    assert len(calls[0]["forbidden_claims"]) == 1


# ---------------------------------------------------------------------------
# Legacy escape hatch: comparison against the pre-semantic scorer must stay
# available and must never touch the judge.
# ---------------------------------------------------------------------------


def test_legacy_mode_disables_the_fallback_via_param_and_env_var(monkeypatch):
    scenario = _grounding_scenario()
    trace = _trace("I rebooked you on the next flight at no charge.")

    def failing_if_called(spec, messages, max_output_tokens, temperature, timeout_seconds):
        raise AssertionError("the semantic judge must not be called in legacy mode")

    via_param = check_factual_grounding(trace, scenario, mode="legacy", caller=failing_if_called)
    assert via_param["required_passed"] is False
    assert via_param["grounding_mode"] == "legacy"
    assert "semantic_fallback" not in via_param

    monkeypatch.setenv("OPENVOICECS_GROUNDING_MODE", "legacy")
    via_env = check_factual_grounding(trace, scenario, caller=failing_if_called)
    assert via_env["required_passed"] is False


# ---------------------------------------------------------------------------
# Judge configuration: pinned defaults, env override, temperature 0.
# ---------------------------------------------------------------------------


def test_default_grounding_judge_spec_is_pinned_and_env_overridable(monkeypatch):
    monkeypatch.delenv("OPENVOICECS_GROUNDING_JUDGE", raising=False)
    default_spec = _default_grounding_judge_spec()
    assert default_spec.provider == DEFAULT_GROUNDING_JUDGE_PROVIDER
    assert default_spec.model_id == DEFAULT_GROUNDING_JUDGE_MODEL_ID

    monkeypatch.setenv("OPENVOICECS_GROUNDING_JUDGE", "openrouter:some-org/some-model")
    overridden_spec = _default_grounding_judge_spec()
    assert overridden_spec.provider == "openrouter"
    assert overridden_spec.model_id == "some-org/some-model"


def test_semantic_fallback_calls_the_judge_with_temperature_zero():
    scenario = _grounding_scenario()
    trace = _trace("I rebooked you on the next flight at no charge.")

    seen = {}

    def caller(spec, messages, max_output_tokens, temperature, timeout_seconds):
        del messages, max_output_tokens, timeout_seconds
        seen["temperature"] = temperature
        seen["model_id"] = spec.model_id
        return json.dumps({
            "required_claims": [{"id": "fee_waived", "grounded": True, "reason": "stub"}],
            "forbidden_claims": [],
        })

    check_factual_grounding(trace, scenario, caller=caller)
    assert seen["temperature"] == 0.0
    assert seen["model_id"] == DEFAULT_GROUNDING_JUDGE_MODEL_ID


# ---------------------------------------------------------------------------
# A judge-call failure is a measurement gap, not evidence the agent under
# test hallucinated -- it must classify as infrastructure, matching the same
# infra-exclusion contract tests/unit/test_scoring_validity.py locks for
# provider errors on the agent side.
# ---------------------------------------------------------------------------


def test_semantic_caller_failure_is_wrapped_and_classified_as_infrastructure():
    scenario = _grounding_scenario()
    trace = _trace("This reply does not literally match any required phrase.")

    def broken_caller(spec, messages, max_output_tokens, temperature, timeout_seconds):
        raise RuntimeError("connection reset by peer")

    with pytest.raises(RuntimeError, match="grounding judge call failed"):
        check_factual_grounding(trace, scenario, caller=broken_caller)

    try:
        check_factual_grounding(trace, scenario, caller=broken_caller)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert classify_trial_error(str(exc)) == "infrastructure"


def test_grounding_judge_failure_excludes_the_trial_instead_of_scoring_it_zero(monkeypatch):
    bench = OpenVoiceCSBench.load()
    single = OpenVoiceCSBench(scenarios=[bench.scenarios[0]])

    def always_broken(trace, scenario):
        del trace, scenario
        raise RuntimeError("grounding judge call failed: connection reset by peer")

    monkeypatch.setattr(
        "src.evaluation.benchmark.openvoicecs.check_factual_grounding",
        always_broken,
    )

    report = single.score_agent(oracle_agent, trials=1)
    result = report["results"][0]
    trial = result["trials"][0]

    assert trial["error_class"] == "infrastructure"
    assert result["measured"] is False
    assert report["num_measured_scenarios"] == 0
    assert report["measurement_coverage"]["infrastructure_error_trials"] == 1
