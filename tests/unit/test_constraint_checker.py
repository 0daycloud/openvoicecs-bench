"""Adversarial trajectory tests for the constraint-based checker.

Each trajectory is a hand-written static fixture, not a model output — these
exist to prove the checker rewards correct-but-reordered trajectories and
rejects incomplete, wrong-decision, and forbidden-call trajectories for the
right reason.
"""

from __future__ import annotations

from copy import deepcopy

from src.evaluation.benchmark.constraint_checker import (
    CallPattern,
    ForbiddenCall,
    GradingSpec,
    OrderingConstraint,
    check_trajectory,
)
from src.evaluation.benchmark.openvoicecs import OpenVoiceCSBench, oracle_agent

REAL_SCENARIO_ID = "retail-refund-damaged-item-001"
REAL_GRADING_SPEC = {
    "required_calls": [
        {"name": "verify_identity", "arguments": {"account_id": "acct_1001"}},
        {"name": "issue_refund", "arguments": {"order_id": "ord_7001", "amount_cents": 5299}},
        {"name": "create_case", "arguments": {"account_id": "acct_1001"}},
    ],
    "ordering_constraints": [
        {"before": "verify_identity", "after": "issue_refund"},
        {"before": "verify_identity", "after": "create_case"},
    ],
    "forbidden_calls": [
        {"name": "issue_refund", "requires_prior_call": "verify_identity"},
    ],
    "expected_outcome": {
        "orders.ord_7001.refund_status": "issued",
        "cases.case_9001.status": "closed_refund_issued",
    },
}


def _real_bench_with_grading_spec() -> OpenVoiceCSBench:
    base = OpenVoiceCSBench.load()
    scenario = deepcopy(next(s for s in base.scenarios if s["id"] == REAL_SCENARIO_ID))
    scenario["oracle"]["grading_spec"] = REAL_GRADING_SPEC
    return OpenVoiceCSBench(scenarios=[scenario], metadata=base.metadata, version=base.version)


REFUND_SPEC = GradingSpec(
    required_calls=[
        CallPattern("verify_identity", {"account_id": "acct_1"}),
        CallPattern("eligibility_check", {"order_id": "ord_1"}),
        CallPattern("process_refund", {"order_id": "ord_1", "amount": 52.99}),
        CallPattern("create_case", {"reason": "damaged_item"}),
    ],
    ordering_constraints=[
        OrderingConstraint(before="verify_identity", after="process_refund"),
        OrderingConstraint(before="eligibility_check", after="process_refund"),
    ],
    forbidden_calls=[
        ForbiddenCall(name="process_refund", requires_prior_call="eligibility_check"),
    ],
    expected_outcome={"refund_approved": True, "amount": 52.99},
    optional_calls=[CallPattern("send_confirmation_email")],
)


def _call(name: str, **arguments) -> dict:
    return {"name": name, "arguments": arguments}


# ---------------------------------------------------------------------------
# Correct but reordered: independent calls in a different sequence must PASS.
# ---------------------------------------------------------------------------


def test_canonical_order_passes():
    trajectory = {
        "tool_calls": [
            _call("verify_identity", account_id="acct_1"),
            _call("eligibility_check", order_id="ord_1"),
            _call("process_refund", order_id="ord_1", amount=52.99),
            _call("create_case", reason="damaged_item"),
        ],
        "outcome": {"refund_approved": True, "amount": 52.99},
    }
    result = check_trajectory(trajectory, REFUND_SPEC)
    assert result["passed"] is True


def test_swapped_independent_calls_still_pass():
    """eligibility_check and verify_identity have no ordering constraint between them."""
    trajectory = {
        "tool_calls": [
            _call("eligibility_check", order_id="ord_1"),
            _call("verify_identity", account_id="acct_1"),
            _call("process_refund", order_id="ord_1", amount=52.99),
            _call("create_case", reason="damaged_item"),
        ],
        "outcome": {"refund_approved": True, "amount": 52.99},
    }
    result = check_trajectory(trajectory, REFUND_SPEC)
    assert result["passed"] is True


def test_optional_call_in_different_position_still_passes():
    trajectory = {
        "tool_calls": [
            _call("verify_identity", account_id="acct_1"),
            _call("eligibility_check", order_id="ord_1"),
            _call("send_confirmation_email"),
            _call("process_refund", order_id="ord_1", amount=52.99),
            _call("create_case", reason="damaged_item"),
        ],
        "outcome": {"refund_approved": True, "amount": 52.99},
    }
    result = check_trajectory(trajectory, REFUND_SPEC)
    assert result["passed"] is True
    assert result["optional_calls"]["occurred"] == ["send_confirmation_email"]


# ---------------------------------------------------------------------------
# Missing steps: FAIL with the missing step named.
# ---------------------------------------------------------------------------


def test_missing_ticket_creation_fails():
    trajectory = {
        "tool_calls": [
            _call("verify_identity", account_id="acct_1"),
            _call("eligibility_check", order_id="ord_1"),
            _call("process_refund", order_id="ord_1", amount=52.99),
        ],
        "outcome": {"refund_approved": True, "amount": 52.99},
    }
    result = check_trajectory(trajectory, REFUND_SPEC)
    assert result["passed"] is False
    assert result["required_calls"]["passed"] is False
    assert result["required_calls"]["missing"][0]["name"] == "create_case"


def test_right_tool_wrong_argument_counts_as_missing():
    """Argument matching is exact, not fuzzy -- a wrong value is a miss, not a near-pass."""
    trajectory = {
        "tool_calls": [
            _call("verify_identity", account_id="acct_1"),
            _call("eligibility_check", order_id="ord_1"),
            _call("process_refund", order_id="ord_1", amount=39.99),  # wrong amount
            _call("create_case", reason="damaged_item"),
        ],
        "outcome": {"refund_approved": True, "amount": 52.99},
    }
    result = check_trajectory(trajectory, REFUND_SPEC)
    assert result["passed"] is False
    assert result["required_calls"]["missing"][0]["name"] == "process_refund"


def test_missing_identity_verification_fails():
    trajectory = {
        "tool_calls": [
            _call("eligibility_check", order_id="ord_1"),
            _call("process_refund", order_id="ord_1", amount=52.99),
            _call("create_case", reason="damaged_item"),
        ],
        "outcome": {"refund_approved": True, "amount": 52.99},
    }
    result = check_trajectory(trajectory, REFUND_SPEC)
    assert result["passed"] is False
    assert result["required_calls"]["missing"][0]["name"] == "verify_identity"


def test_missing_eligibility_check_fails_required_and_forbidden():
    """Missing eligibility_check is both a required-call gap and a forbidden refund."""
    trajectory = {
        "tool_calls": [
            _call("verify_identity", account_id="acct_1"),
            _call("process_refund", order_id="ord_1", amount=52.99),
            _call("create_case", reason="damaged_item"),
        ],
        "outcome": {"refund_approved": True, "amount": 52.99},
    }
    result = check_trajectory(trajectory, REFUND_SPEC)
    assert result["passed"] is False
    assert result["required_calls"]["missing"][0]["name"] == "eligibility_check"
    assert result["forbidden_calls"]["passed"] is False
    assert result["forbidden_calls"]["violations"][0]["name"] == "process_refund"


# ---------------------------------------------------------------------------
# Wrong decisions: right calls, wrong ground-truth outcome.
# ---------------------------------------------------------------------------


def test_approving_ineligible_refund_fails_outcome_check():
    ineligible_spec = GradingSpec(
        required_calls=REFUND_SPEC.required_calls,
        ordering_constraints=REFUND_SPEC.ordering_constraints,
        forbidden_calls=REFUND_SPEC.forbidden_calls,
        expected_outcome={"refund_approved": False},
    )
    trajectory = {
        "tool_calls": [
            _call("verify_identity", account_id="acct_1"),
            _call("eligibility_check", order_id="ord_1"),
            _call("process_refund", order_id="ord_1", amount=52.99),
            _call("create_case", reason="damaged_item"),
        ],
        "outcome": {"refund_approved": True, "amount": 52.99},
    }
    result = check_trajectory(trajectory, ineligible_spec)
    assert result["passed"] is False
    assert result["expected_outcome"]["mismatches"][0]["key"] == "refund_approved"


def test_rejecting_eligible_refund_fails_outcome_check():
    trajectory = {
        "tool_calls": [
            _call("verify_identity", account_id="acct_1"),
            _call("eligibility_check", order_id="ord_1"),
            _call("create_case", reason="damaged_item"),
        ],
        "outcome": {"refund_approved": False},
    }
    result = check_trajectory(trajectory, REFUND_SPEC)
    assert result["passed"] is False
    assert result["expected_outcome"]["mismatches"][0]["key"] == "refund_approved"
    assert result["expected_outcome"]["mismatches"][0]["expected"] is True


# ---------------------------------------------------------------------------
# Forbidden calls: acting before the gate is satisfied.
# ---------------------------------------------------------------------------


def test_refund_before_eligibility_check_is_forbidden():
    trajectory = {
        "tool_calls": [
            _call("verify_identity", account_id="acct_1"),
            _call("process_refund", order_id="ord_1", amount=52.99),
            _call("eligibility_check", order_id="ord_1"),
            _call("create_case", reason="damaged_item"),
        ],
        "outcome": {"refund_approved": True, "amount": 52.99},
    }
    result = check_trajectory(trajectory, REFUND_SPEC)
    assert result["passed"] is False
    assert result["forbidden_calls"]["passed"] is False
    assert "eligibility_check" in result["forbidden_calls"]["violations"][0]["reason"]
    assert result["ordering_constraints"]["passed"] is False


# ---------------------------------------------------------------------------
# Wired into the real pipeline: a real corpus scenario, real replay, real
# score_agent -- proves oracle.grading_spec is picked up end to end and that
# scenarios without it are completely unaffected.
# ---------------------------------------------------------------------------


def test_real_scenario_reordered_independent_calls_pass_via_score_agent():
    bench = _real_bench_with_grading_spec()

    def reordered_agent(scenario, trial_index):
        return {"tool_calls": [
            {"name": "verify_identity", "arguments": {"account_id": "acct_1001"}},
            {"name": "create_case", "arguments": {"account_id": "acct_1001", "reason": "damaged_item"}},
            {
                "name": "issue_refund",
                "arguments": {"order_id": "ord_7001", "amount_cents": 5299, "reason": "damaged_item"},
            },
        ]}

    report = bench.score_agent(reordered_agent, trials=1)
    trial = report["results"][0]["trials"][0]
    assert trial["constraint_check"]["passed"] is True


def test_real_scenario_missing_step_fails_via_score_agent():
    bench = _real_bench_with_grading_spec()

    def incomplete_agent(scenario, trial_index):
        return {"tool_calls": [
            {"name": "verify_identity", "arguments": {"account_id": "acct_1001"}},
            {
                "name": "issue_refund",
                "arguments": {"order_id": "ord_7001", "amount_cents": 5299, "reason": "damaged_item"},
            },
        ]}

    report = bench.score_agent(incomplete_agent, trials=1)
    trial = report["results"][0]["trials"][0]
    assert trial["constraint_check"]["passed"] is False
    assert trial["constraint_check"]["required_calls"]["missing"][0]["name"] == "create_case"


def test_scenario_without_grading_spec_has_no_constraint_check():
    bench = OpenVoiceCSBench.load()
    report = bench.score_agent(oracle_agent, max_scenarios=1)
    trial = report["results"][0]["trials"][0]
    assert trial["constraint_check"] is None
