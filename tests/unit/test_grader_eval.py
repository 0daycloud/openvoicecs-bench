"""Tests for the harness that grades the grader.

A harness that measures grading errors is only useful if its own expectations
are sound. Three ways it could be wrong, each covered below:

1. a mutation that silently changes nothing would report the grader as correct
   about a trace it never saw;
2. a mutation expecting a pass could break the task it claims to preserve, so a
   correct grader would be recorded as producing a false fail;
3. a verdict could be assigned by anything other than the expectation.
"""

from __future__ import annotations

from src.evaluation.benchmark.grader_eval import (
    MUTATIONS,
    Mutation,
    _base_trace,
    _classify,
    _set_text,
    evaluate_grader,
    unreachable_forbidden_patterns,
)
from src.evaluation.benchmark.openvoicecs import OpenVoiceCSBench, is_multi_turn

# Scoring every mutation over 220 scenarios is a minute of work. These cover a
# refund with a bookkeeping call, a fee waiver, a multi-turn call, and an
# adversarial refusal — every mutation finds a home in at least one.
SAMPLE_IDS = (
    "retail-refund-damaged-item-001",
    "travel-rebook-missed-connection-001",
    "retail-multiturn-verify-then-refund-912",
    "healthcare-phi-extraction-001",
)


def _sample_suite() -> OpenVoiceCSBench:
    suite = OpenVoiceCSBench.load()
    scenarios = [s for s in suite.scenarios if s["id"] in SAMPLE_IDS]
    assert len(scenarios) == len(SAMPLE_IDS), "sample scenario ids drifted out of the corpus"
    return OpenVoiceCSBench(scenarios=scenarios, version=suite.version)


def test_the_events_channel_is_closed_so_derivation_is_actually_exercised():
    """The oracle self-reports its required events; fabricated traces must not.

    ``provider_adapters`` gives a model no events channel, so a trace that
    declares its own events measures behaviour no model can reproduce. If this
    ever regresses, every event-derived metric passes vacuously and the harness
    stops measuring the thing it exists for.
    """
    suite = OpenVoiceCSBench.load()
    for scenario in suite.scenarios[:20]:
        assert _base_trace(scenario)["events"] == []


def test_every_mutation_changes_the_trace_where_it_applies():
    """A mutation that returns an unchanged trace would grade the control twice."""
    suite = OpenVoiceCSBench.load()
    applied: dict[str, int] = {}
    for mutation in MUTATIONS:
        if mutation.id == "control":
            continue
        for scenario in suite.scenarios:
            base = _base_trace(scenario)
            mutated = mutation.build(scenario, _base_trace(scenario))
            if mutated is None:
                continue
            applied[mutation.id] = applied.get(mutation.id, 0) + 1
            assert (mutated["tool_calls"], mutated["messages"]) != (
                base["tool_calls"],
                base["messages"],
            ), f"{mutation.id} left {scenario['id']} unchanged"

    for mutation in MUTATIONS:
        if mutation.id == "control":
            continue
        assert applied.get(mutation.id), f"{mutation.id} never applied to any scenario"


def test_mutations_expecting_a_pass_keep_every_required_tool_call():
    """A pass-expecting mutation must not remove work the oracle required.

    Without this, a mutation could break the task and the resulting failure
    would be booked as a grader false fail — an error the grader did not make.
    """
    suite = OpenVoiceCSBench.load()
    for mutation in MUTATIONS:
        if not mutation.should_pass or mutation.id == "control":
            continue
        for scenario in suite.scenarios:
            mutated = mutation.build(scenario, _base_trace(scenario))
            if mutated is None:
                continue
            before = sorted(call["name"] for call in _base_trace(scenario)["tool_calls"])
            after = sorted(call["name"] for call in mutated["tool_calls"])
            assert before == after, f"{mutation.id} changed the calls for {scenario['id']}"


def test_verdicts_follow_the_expectation_not_the_outcome():
    passing = {"pass_at_k": True, "avg_scores": {"safety": 1.0}}
    failing = {"pass_at_k": False, "avg_scores": {"safety": 0.0}}
    failing_elsewhere = {"pass_at_k": False, "avg_scores": {"safety": 1.0}}

    expects_pass = Mutation("m", "", True, None, lambda s, t: t)
    expects_fail = Mutation("m", "", False, "safety", lambda s, t: t)

    assert _classify(expects_pass, passing) == "correct"
    assert _classify(expects_pass, failing) == "false_fail"
    assert _classify(expects_fail, failing) == "correct"
    assert _classify(expects_fail, passing) == "false_pass"
    # Failing for an unrelated reason is not a success: the diagnosis a user
    # would read points at the wrong metric.
    assert _classify(expects_fail, failing_elsewhere) == "wrong_metric"


def test_unreachable_forbidden_patterns_separates_reachable_from_neutralised():
    """A forbidden pattern keyed on a generated argument can never match.

    Replay substitutes the tool's declared value for anything the model sent, so
    the call stops being the forbidden one. The retail refund scenario's
    forbidden replacement is keyed on ``order_id``, which the model does control,
    and must stay reachable.
    """
    suite = OpenVoiceCSBench.load()
    reachability = unreachable_forbidden_patterns(suite)

    assert reachability["total_patterns"] > reachability["unreachable_patterns"] > 0
    affected = {example["scenario_id"] for example in reachability["examples"]}
    assert "retail-refund-damaged-item-001" not in affected


def test_a_scenario_counts_as_mutated_when_any_turn_took_the_mutation():
    """Multi-turn scenarios must not be dropped because a later turn had nothing to do.

    The oracle issues only the calls still outstanding, so work done on turn one
    is simply absent from turn two. Disqualifying the scenario there would
    silently remove multi-turn cases from the measurement they matter most to.
    """
    calls: list[int] = []

    def only_the_first_turn(scenario: dict, trace: dict) -> dict | None:
        calls.append(1)
        if len(calls) > 1:
            return None
        return _set_text(trace, "Rewritten on the first turn only.")

    suite = OpenVoiceCSBench.load()
    multi_turn = next(s for s in suite.scenarios if is_multi_turn(s))
    sub = OpenVoiceCSBench(scenarios=[multi_turn], version=suite.version)

    report = evaluate_grader(
        sub, mutations=(Mutation("first_turn_only", "", True, None, only_the_first_turn),)
    )
    assert len(calls) > 1, "picked a scenario the harness only calls once"
    assert report["by_mutation"]["first_turn_only"]["applied"] == 1


def test_report_is_deterministic():
    suite = _sample_suite()
    first = evaluate_grader(suite)
    second = evaluate_grader(suite)
    assert first["by_mutation"] == second["by_mutation"]
    assert first["cases_applied"] == second["cases_applied"]


def test_report_counts_add_up():
    report = evaluate_grader(_sample_suite())
    for name, counts in report["by_mutation"].items():
        total = (
            counts["correct"] + counts["false_pass"] + counts["false_fail"]
            + counts["wrong_metric"]
        )
        assert total == counts["applied"], f"{name} verdicts do not sum to cases applied"
