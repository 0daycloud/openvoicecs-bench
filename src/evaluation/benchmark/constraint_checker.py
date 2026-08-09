"""Deterministic, order-tolerant constraint checker for agent trajectories.

The existing OpenVoiceCS oracle (``openvoicecs.py``) already replays tool
calls against scenario state and is set-based rather than positional. This
module adds a lighter-weight, explicit alternative for scenarios that would
rather declare their contract directly — required calls, causal ordering
between two specific calls, calls that are forbidden unless a prior call
happened, and a ground-truth outcome — without needing a full replayable
state machine. It never asks a model to judge correctness; every check is a
structural comparison against the spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CallPattern:
    """A tool call to match by name and an argument subset."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderingConstraint:
    """``before`` must be called at least once before the first ``after`` call."""

    before: str
    after: str


@dataclass(frozen=True)
class ForbiddenCall:
    """A call that fails the trajectory, optionally only absent a prior call.

    If ``requires_prior_call`` is ``None`` the call is forbidden outright. If
    set, the call is only a violation when it appears without that prior call
    name occurring earlier in the trajectory (e.g. ``process_refund`` without
    an earlier ``eligibility_check``).
    """

    name: str
    requires_prior_call: str | None = None


@dataclass(frozen=True)
class GradingSpec:
    """A causal, order-tolerant grading contract for one scenario."""

    required_calls: list[CallPattern] = field(default_factory=list)
    ordering_constraints: list[OrderingConstraint] = field(default_factory=list)
    forbidden_calls: list[ForbiddenCall] = field(default_factory=list)
    expected_outcome: dict[str, Any] = field(default_factory=dict)
    optional_calls: list[CallPattern] = field(default_factory=list)


def spec_from_dict(data: dict[str, Any]) -> GradingSpec:
    """Build a ``GradingSpec`` from the plain-dict shape scenarios author it in."""
    return GradingSpec(
        required_calls=[_pattern_from_dict(item) for item in data.get("required_calls", [])],
        ordering_constraints=[
            OrderingConstraint(before=item["before"], after=item["after"])
            for item in data.get("ordering_constraints", [])
        ],
        forbidden_calls=[
            ForbiddenCall(name=item["name"], requires_prior_call=item.get("requires_prior_call"))
            for item in data.get("forbidden_calls", [])
        ],
        expected_outcome=dict(data.get("expected_outcome", {})),
        optional_calls=[_pattern_from_dict(item) for item in data.get("optional_calls", [])],
    )


def _pattern_from_dict(item: dict[str, Any]) -> CallPattern:
    return CallPattern(name=item["name"], arguments=dict(item.get("arguments", {})))


def check_trajectory(trajectory: dict[str, Any], spec: GradingSpec) -> dict[str, Any]:
    """Score a trajectory against a ``GradingSpec``.

    ``trajectory`` is ``{"tool_calls": [{"name": ..., "arguments": {...}}, ...],
    "outcome": {...}}``. Returns a breakdown of every check rather than a
    single score, so a failure is debuggable without re-running anything.
    """
    calls = trajectory.get("tool_calls") or []
    outcome = trajectory.get("outcome") or {}

    required = _check_required_calls(calls, spec.required_calls)
    ordering = _check_ordering_constraints(calls, spec.ordering_constraints)
    forbidden = _check_forbidden_calls(calls, spec.forbidden_calls)
    outcome_check = _check_expected_outcome(outcome, spec.expected_outcome)
    optional = _check_optional_calls(calls, spec.optional_calls)

    passed = (
        required["passed"]
        and ordering["passed"]
        and forbidden["passed"]
        and outcome_check["passed"]
    )
    return {
        "passed": passed,
        "required_calls": required,
        "ordering_constraints": ordering,
        "forbidden_calls": forbidden,
        "expected_outcome": outcome_check,
        "optional_calls": optional,
    }


def _matches(call: dict[str, Any], pattern: CallPattern) -> bool:
    if call.get("name") != pattern.name:
        return False
    return _dict_subset(call.get("arguments") or {}, pattern.arguments)


def _dict_subset(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if key not in actual:
            return False
        actual_value = actual[key]
        if isinstance(value, dict) and isinstance(actual_value, dict):
            if not _dict_subset(actual_value, value):
                return False
        elif actual_value != value:
            return False
    return True


def _first_index(calls: list[dict[str, Any]], name: str) -> int | None:
    for index, call in enumerate(calls):
        if call.get("name") == name:
            return index
    return None


def _check_required_calls(
    calls: list[dict[str, Any]],
    patterns: list[CallPattern],
) -> dict[str, Any]:
    missing = [
        {"name": pattern.name, "arguments": pattern.arguments}
        for pattern in patterns
        if not any(_matches(call, pattern) for call in calls)
    ]
    return {"passed": not missing, "missing": missing}


def _check_ordering_constraints(
    calls: list[dict[str, Any]],
    constraints: list[OrderingConstraint],
) -> dict[str, Any]:
    violations = []
    for constraint in constraints:
        before_index = _first_index(calls, constraint.before)
        after_index = _first_index(calls, constraint.after)
        if after_index is None:
            continue
        if before_index is None or before_index > after_index:
            violations.append({
                "before": constraint.before,
                "after": constraint.after,
                "reason": (
                    f"'{constraint.before}' never occurred before '{constraint.after}'"
                    if before_index is None
                    else f"'{constraint.before}' occurred at index {before_index}, "
                    f"after '{constraint.after}' at index {after_index}"
                ),
            })
    return {"passed": not violations, "violations": violations}


def _check_forbidden_calls(
    calls: list[dict[str, Any]],
    forbidden: list[ForbiddenCall],
) -> dict[str, Any]:
    violations = []
    for rule in forbidden:
        for index, call in enumerate(calls):
            if call.get("name") != rule.name:
                continue
            if rule.requires_prior_call is None:
                violations.append({"name": rule.name, "index": index, "reason": "call is unconditionally forbidden"})
                continue
            prior_index = _first_index(calls[:index], rule.requires_prior_call)
            if prior_index is None:
                violations.append({
                    "name": rule.name,
                    "index": index,
                    "reason": f"called without a prior '{rule.requires_prior_call}'",
                })
    return {"passed": not violations, "violations": violations}


def _get_path(data: dict[str, Any], dotted_key: str) -> Any:
    """Look up a possibly dot-separated key, e.g. ``orders.ord_1.status``."""
    current: Any = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _check_expected_outcome(
    outcome: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    mismatches = [
        {"key": key, "expected": value, "actual": _get_path(outcome, key)}
        for key, value in expected.items()
        if _get_path(outcome, key) != value
    ]
    return {"passed": not mismatches, "mismatches": mismatches}


def _check_optional_calls(
    calls: list[dict[str, Any]],
    patterns: list[CallPattern],
) -> dict[str, Any]:
    return {
        "occurred": [pattern.name for pattern in patterns if any(_matches(call, pattern) for call in calls)],
        "not_occurred": [
            pattern.name for pattern in patterns if not any(_matches(call, pattern) for call in calls)
        ],
    }
