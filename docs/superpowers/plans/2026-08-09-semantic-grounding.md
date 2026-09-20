# Semantic Factual-Grounding Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, offline, blinded semantic grounding check for OpenVoiceCS-Bench that classifies each required claim as `grounded` / `honest_alternative` / `not_grounded` via a narrow LLM-judge call, fixing the synonymy-miss and honest-failure-penalization bugs in the literal phrase matcher (`docs/known-limitations.md` section 7) without touching the existing matcher or `overall_score`.

**Architecture:** New module `src/evaluation/benchmark/semantic_grounding.py`, structured like the existing model-judge pipeline in `judging.py`: blinded item extraction → narrow judge calls → aggregation → additive merge into a saved report. Reuses `ModelJudgeSpec`, `call_openai_compatible_model_judge`, `parse_model_judge_spec`, `JudgeIssue` from `judging.py` rather than duplicating them. Two new CLI subcommands in `scripts/run_openvoicecs.py` mirror `model-judge` / `apply-judge-report`.

**Tech Stack:** Python 3.10+, pytest, existing OpenVoiceCS-Bench harness (`src/evaluation/benchmark/*`). No new dependencies.

## Global Constraints

- Full design: `docs/superpowers/specs/2026-08-09-semantic-grounding-design.md` — read it before starting; every task below implements one piece of it.
- Do not modify `check_factual_grounding`, `metric_scores`, or `overall_score` computation anywhere. This is additive only.
- No changes to `data/openvoicecs/scenarios_*.json` or any release artifact.
- Tests must not require a live API key or network call — every judge call in tests uses a fake `caller` closure (same pattern as `tests/unit/test_judging.py`).
- Follow existing style: `from __future__ import annotations`, dataclasses/types as seen in `judging.py`, double-quoted strings, existing import-sorting convention (stdlib, then `src.*` alphabetically).
- Line length: this repo carries pre-existing `E501` debt and doesn't gate lint on it, but don't add new long lines gratuitously — wrap like the surrounding code does.

---

### Task 1: Blinded item extraction

**Files:**
- Create: `src/evaluation/benchmark/semantic_grounding.py`
- Test: `tests/unit/test_semantic_grounding.py`

**Interfaces:**
- Consumes: `ModelJudgeSpec`, `ModelJudgeCaller`, `JudgeIssue`, `call_openai_compatible_model_judge`, `parse_model_judge_spec`, `_extract_json_object`, `_trial_index_from_report` from `src.evaluation.benchmark.judging`. `OpenVoiceCSBench`, `validate_report` from `src.evaluation.benchmark.openvoicecs` (test only).
- Produces: `GROUNDING_VERDICTS: tuple[str, ...]`, `SATISFIED_VERDICTS: frozenset[str]`, `iter_blinded_grounding_items(report: dict, scenarios: list[dict]) -> list[dict]`. Every item has keys `item_id, scenario_id, trial_index, claim_id, claim_description, agent_text`. Later tasks depend on exactly these keys and on `trial_index` being an `int`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_semantic_grounding.py`:

```python
"""Tests for the opt-in semantic factual-grounding check."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from src.evaluation.benchmark.judging import ModelJudgeSpec
from src.evaluation.benchmark.openvoicecs import OpenVoiceCSBench, validate_report
from src.evaluation.benchmark.semantic_grounding import iter_blinded_grounding_items


def _grounding_scenario() -> dict:
    return {
        "id": "grounding-test-001",
        "domain": "billing",
        "track": "text_to_action",
        "difficulty": "easy",
        "customer_goal": "Get the modem rental fee waived.",
        "conversation": [
            {"role": "customer", "text": "Can you waive the modem rental fee?"}
        ],
        "initial_state": {"accounts": {"acct_1": {}}},
        "tools": [],
        "oracle": {
            "expected_state": {},
            "grounding": {
                "required_claims": [
                    {
                        "id": "fee_waived",
                        "any_terms": ["no change fee", "no fee", "fee waiver"],
                    }
                ],
            },
        },
    }


def _agent_with_text(text: str):
    def agent_fn(scenario, trial_index):
        del scenario, trial_index
        return {"messages": [{"role": "agent", "text": text}]}

    return agent_fn


def test_iter_blinded_grounding_items_returns_one_item_per_required_claim():
    scenario = _grounding_scenario()
    report = OpenVoiceCSBench(scenarios=[scenario]).score_agent(
        _agent_with_text(
            "I'm sorry, I can't waive that fee myself, so I've escalated it to billing."
        ),
        trials=1,
    )

    items = iter_blinded_grounding_items(report, [scenario])

    assert len(items) == 1
    item = items[0]
    assert item["item_id"] == "grounding-test-001:0:fee_waived"
    assert item["scenario_id"] == "grounding-test-001"
    assert item["trial_index"] == 0
    assert item["claim_id"] == "fee_waived"
    assert item["claim_description"] == "no change fee / no fee / fee waiver"
    assert "escalated" in item["agent_text"]
    assert set(item) == {
        "item_id", "scenario_id", "trial_index", "claim_id",
        "claim_description", "agent_text",
    }


def test_iter_blinded_grounding_items_skips_scenarios_without_required_claims():
    scenario = _grounding_scenario()
    scenario["id"] = "no-claims-001"
    del scenario["oracle"]["grounding"]
    report = OpenVoiceCSBench(scenarios=[scenario]).score_agent(
        _agent_with_text("Sure, done."), trials=1
    )

    assert iter_blinded_grounding_items(report, [scenario]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_semantic_grounding.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.evaluation.benchmark.semantic_grounding'`

- [ ] **Step 3: Write the implementation**

Create `src/evaluation/benchmark/semantic_grounding.py`:

```python
"""Offline, blinded semantic grounding check for OpenVoiceCS required claims.

Deterministic phrase-matched grounding (``check_factual_grounding`` in
openvoicecs.py) remains the default and only score that feeds
``metric_scores.factual_grounding`` / ``overall_score``. This module adds a
separate, auditable path: narrow LLM-judge calls that classify each required
claim as grounded, honestly addressed by a different truthful outcome, or
genuinely missing. It follows the same blinded/audited pattern as the
model-judge pipeline in judging.py, run offline and post-hoc against an
already-scored report rather than inline during scoring.
"""

from __future__ import annotations

from typing import Any

from src.evaluation.benchmark.judging import _trial_index_from_report

GROUNDING_VERDICTS = ("grounded", "honest_alternative", "not_grounded")
SATISFIED_VERDICTS = frozenset({"grounded", "honest_alternative"})


def iter_blinded_grounding_items(
    report: dict[str, Any],
    scenarios: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return one blinded item per (trial, required claim).

    ``report["results"]`` does not carry ``oracle.grounding.required_claims``
    -- only the literal matcher's ``missing_required_claims`` is present in
    each trial's ``grounding_check``, which is not the full claim set an
    independent judge needs. ``scenarios`` is the same suite the report was
    produced from (e.g. ``OpenVoiceCSBench.load(path).scenarios``), matched
    to report results by ``id`` -- stable for audio variants too, since
    ``build_audio_variant_scenarios`` copies the base scenario's oracle
    under the variant's own id.

    Items carry no scenario identity beyond the id needed to key results
    back together, no model identity, no oracle pass/fail, no tool calls,
    and no scores -- only the claim text and the agent's own words.
    """
    scenarios_by_id = {
        str(scenario.get("id")): scenario
        for scenario in scenarios
        if isinstance(scenario, dict) and scenario.get("id")
    }
    items: list[dict[str, Any]] = []
    for result in report.get("results", []):
        if not isinstance(result, dict):
            continue
        scenario_id = str(result.get("id") or "")
        scenario = scenarios_by_id.get(scenario_id)
        if scenario is None:
            continue
        required_claims = _required_claims(scenario)
        if not required_claims:
            continue
        trials = result.get("trials") if isinstance(result.get("trials"), list) else []
        for fallback_index, trial in enumerate(trials):
            if not isinstance(trial, dict):
                continue
            trial_index = _trial_index_from_report(trial, fallback_index)
            agent_text = _blinded_agent_text(trial.get("messages"))
            if not agent_text:
                continue
            for claim in required_claims:
                claim_id = str(claim.get("id") or "")
                if not claim_id:
                    continue
                items.append({
                    "item_id": f"{scenario_id}:{trial_index}:{claim_id}",
                    "scenario_id": scenario_id,
                    "trial_index": trial_index,
                    "claim_id": claim_id,
                    "claim_description": _claim_description(claim),
                    "agent_text": agent_text,
                })
    return items


def _required_claims(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    oracle = scenario.get("oracle")
    if not isinstance(oracle, dict):
        return []
    grounding = oracle.get("grounding")
    if not isinstance(grounding, dict):
        return []
    claims = grounding.get("required_claims")
    return claims if isinstance(claims, list) else []


def _claim_description(claim: dict[str, Any]) -> str:
    terms = claim.get("any_terms")
    if isinstance(terms, list) and terms:
        return " / ".join(str(term) for term in terms)
    return str(claim.get("id", ""))


def _blinded_agent_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    parts = [
        str(message.get("text", ""))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "agent"
    ]
    return " ".join(part for part in parts if part).strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_semantic_grounding.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/benchmark/semantic_grounding.py tests/unit/test_semantic_grounding.py
git commit -m "Add blinded item extraction for semantic grounding check"
```

---

### Task 2: Narrow LLM-judge calls

**Files:**
- Modify: `src/evaluation/benchmark/semantic_grounding.py`
- Test: `tests/unit/test_semantic_grounding.py`

**Interfaces:**
- Consumes: `iter_blinded_grounding_items`, `GROUNDING_VERDICTS` (Task 1); `ModelJudgeSpec`, `ModelJudgeCaller`, `call_openai_compatible_model_judge`, `_extract_json_object` from `judging.py`.
- Produces: `generate_semantic_grounding_annotations(report, scenarios, *, judge_specs, adjudicator=None, caller=None, max_output_tokens=200, temperature=0.0, timeout_seconds=60.0) -> list[dict]`. Each annotation has keys `item_id, scenario_id, trial_index, claim_id, rater_id, verdict, rationale, judge` where `judge = {"type": "audited_grounding_judge", "provider": str, "model_id": str, "adjudicator": bool}`. Task 3 depends on exactly these keys.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_semantic_grounding.py` (add these imports to the existing import block at the top of the file: `from src.evaluation.benchmark.semantic_grounding import (generate_semantic_grounding_annotations, iter_blinded_grounding_items)` — replace the single-name import from Task 1 with this multi-name one):

```python
def test_generate_semantic_grounding_annotations_blinds_payload_and_returns_verdict():
    scenario = _grounding_scenario()
    report = OpenVoiceCSBench(scenarios=[scenario]).score_agent(
        _agent_with_text(
            "I'm sorry, I can't waive that fee myself, so I've escalated it to billing."
        ),
        trials=1,
    )
    seen_payloads = []

    def caller(spec, messages, max_output_tokens, temperature, timeout_seconds):
        del spec, max_output_tokens, temperature, timeout_seconds
        payload = json.loads(messages[1]["content"])
        seen_payloads.append(payload)
        assert set(payload) == {"claim", "agent_text"}
        return json.dumps({"verdict": "honest_alternative", "rationale": "truthful escalation"})

    annotations = generate_semantic_grounding_annotations(
        report,
        [scenario],
        judge_specs=[
            ModelJudgeSpec(provider="openrouter", model_id="judge-a"),
            ModelJudgeSpec(provider="openrouter", model_id="judge-b"),
        ],
        caller=caller,
    )

    assert len(annotations) == 2
    assert len(seen_payloads) == 2
    assert {a["verdict"] for a in annotations} == {"honest_alternative"}
    assert {a["rater_id"] for a in annotations} == {
        "grounding-judge-openrouter-judge-a",
        "grounding-judge-openrouter-judge-b",
    }
    for annotation in annotations:
        assert annotation["judge"]["type"] == "audited_grounding_judge"
        assert annotation["judge"]["adjudicator"] is False
        assert annotation["claim_id"] == "fee_waived"


def test_generate_semantic_grounding_annotations_calls_adjudicator_on_disagreement():
    scenario = _grounding_scenario()
    report = OpenVoiceCSBench(scenarios=[scenario]).score_agent(
        _agent_with_text("I've escalated this."), trials=1
    )
    calls = []

    def caller(spec, messages, max_output_tokens, temperature, timeout_seconds):
        del messages, max_output_tokens, temperature, timeout_seconds
        calls.append(spec.model_id)
        verdict = {"judge-a": "grounded", "judge-b": "not_grounded"}.get(
            spec.model_id, "honest_alternative"
        )
        return json.dumps({"verdict": verdict})

    annotations = generate_semantic_grounding_annotations(
        report,
        [scenario],
        judge_specs=[
            ModelJudgeSpec(provider="openrouter", model_id="judge-a"),
            ModelJudgeSpec(provider="openrouter", model_id="judge-b"),
        ],
        adjudicator=ModelJudgeSpec(provider="openrouter", model_id="judge-c"),
        caller=caller,
    )

    assert calls == ["judge-a", "judge-b", "judge-c"]
    assert len(annotations) == 3
    assert annotations[-1]["judge"]["adjudicator"] is True


def test_generate_semantic_grounding_annotations_does_not_adjudicate_on_agreement():
    scenario = _grounding_scenario()
    report = OpenVoiceCSBench(scenarios=[scenario]).score_agent(
        _agent_with_text("I've escalated this."), trials=1
    )
    calls = []

    def caller(spec, messages, max_output_tokens, temperature, timeout_seconds):
        del messages, max_output_tokens, temperature, timeout_seconds
        calls.append(spec.model_id)
        return json.dumps({"verdict": "honest_alternative"})

    annotations = generate_semantic_grounding_annotations(
        report,
        [scenario],
        judge_specs=[
            ModelJudgeSpec(provider="openrouter", model_id="judge-a"),
            ModelJudgeSpec(provider="openrouter", model_id="judge-b"),
        ],
        adjudicator=ModelJudgeSpec(provider="openrouter", model_id="judge-c"),
        caller=caller,
    )

    assert calls == ["judge-a", "judge-b"]
    assert len(annotations) == 2


def test_generate_semantic_grounding_annotations_rejects_bad_verdict():
    scenario = _grounding_scenario()
    report = OpenVoiceCSBench(scenarios=[scenario]).score_agent(
        _agent_with_text("Sure thing."), trials=1
    )

    def caller(spec, messages, max_output_tokens, temperature, timeout_seconds):
        del spec, messages, max_output_tokens, temperature, timeout_seconds
        return json.dumps({"verdict": "maybe"})

    with pytest.raises(ValueError, match="must be one of"):
        generate_semantic_grounding_annotations(
            report,
            [scenario],
            judge_specs=[ModelJudgeSpec(provider="openrouter", model_id="judge-a")],
            caller=caller,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_semantic_grounding.py -v`
Expected: FAIL — `ImportError: cannot import name 'generate_semantic_grounding_annotations'`

- [ ] **Step 3: Write the implementation**

Add to `src/evaluation/benchmark/semantic_grounding.py` (insert after the existing imports, extending the `from src.evaluation.benchmark.judging import ...` line, and add module-level constants and functions below `_blinded_agent_text`):

Replace the top of the file, from `from __future__ import annotations` down
through the closing `)` of the `judging` import, with:

```python
from __future__ import annotations

import json
import re
from typing import Any

from src.evaluation.benchmark.judging import (
    ModelJudgeCaller,
    ModelJudgeSpec,
    _extract_json_object,
    _trial_index_from_report,
    call_openai_compatible_model_judge,
)
```

Add after `GROUNDING_VERDICTS` / `SATISFIED_VERDICTS`:

```python
GROUNDING_JUDGE_SYSTEM_PROMPT = (
    "You are checking whether one required claim is grounded in an AI "
    "customer service agent's stated reply.\n\n"
    "Given the claim and the agent's text, respond with exactly one verdict:\n"
    "- grounded: the reply conveys this claim in substance, in any wording.\n"
    "- honest_alternative: the reply does not make this claim, but it "
    "honestly and accurately reports a different outcome (a failure, "
    "inability, or escalation) rather than falsely asserting the claim.\n"
    "- not_grounded: the claim is neither made nor honestly addressed; it "
    "is simply missing.\n\n"
    'Return only JSON: {"verdict": "...", "rationale": "one sentence"}.'
)
```

Add at the end of the file:

```python
def generate_semantic_grounding_annotations(
    report: dict[str, Any],
    scenarios: list[dict[str, Any]],
    *,
    judge_specs: list[ModelJudgeSpec],
    adjudicator: ModelJudgeSpec | None = None,
    caller: ModelJudgeCaller | None = None,
    max_output_tokens: int = 200,
    temperature: float = 0.0,
    timeout_seconds: float = 60.0,
) -> list[dict[str, Any]]:
    """Generate blinded semantic-grounding annotations for every required claim.

    One narrow judge call per (item, judge spec) -- kept to a single claim
    per call rather than batching a trial's claims together, so one
    malformed response invalidates one claim instead of a whole trial.
    """
    if not judge_specs:
        raise ValueError("at least one judge spec is required")
    call = caller or call_openai_compatible_model_judge
    rater_ids = _grounding_rater_ids(judge_specs)
    adjudicator_id = (
        _grounding_rater_id(adjudicator, prefix="adjudicator") if adjudicator else None
    )

    annotations: list[dict[str, Any]] = []
    for item in iter_blinded_grounding_items(report, scenarios):
        item_annotations = []
        for spec, rater_id in zip(judge_specs, rater_ids, strict=True):
            item_annotations.append(
                _score_claim_with_judge(
                    item,
                    spec=spec,
                    rater_id=rater_id,
                    caller=call,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    timeout_seconds=timeout_seconds,
                )
            )
        if (
            adjudicator is not None
            and len(item_annotations) >= 2
            and item_annotations[0]["verdict"] != item_annotations[1]["verdict"]
        ):
            item_annotations.append(
                _score_claim_with_judge(
                    item,
                    spec=adjudicator,
                    rater_id=adjudicator_id or "adjudicator",
                    caller=call,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    timeout_seconds=timeout_seconds,
                    adjudication=True,
                )
            )
        annotations.extend(item_annotations)
    return annotations


def _score_claim_with_judge(
    item: dict[str, Any],
    *,
    spec: ModelJudgeSpec,
    rater_id: str,
    caller: ModelJudgeCaller,
    max_output_tokens: int,
    temperature: float,
    timeout_seconds: float,
    adjudication: bool = False,
) -> dict[str, Any]:
    messages = _build_grounding_judge_messages(item, adjudication=adjudication)
    response_text = caller(spec, messages, max_output_tokens, temperature, timeout_seconds)
    parsed = _parse_grounding_judge_response(response_text)
    return {
        "item_id": item["item_id"],
        "scenario_id": item["scenario_id"],
        "trial_index": item["trial_index"],
        "claim_id": item["claim_id"],
        "rater_id": rater_id,
        "verdict": parsed["verdict"],
        "rationale": parsed["rationale"],
        "judge": {
            "type": "audited_grounding_judge",
            "provider": spec.provider,
            "model_id": spec.model_id,
            "adjudicator": adjudication,
        },
    }


def _build_grounding_judge_messages(
    item: dict[str, Any],
    *,
    adjudication: bool,
) -> list[dict[str, str]]:
    system = GROUNDING_JUDGE_SYSTEM_PROMPT
    if adjudication:
        system += (
            "\nYou are adjudicating a disagreement between two raters. Score "
            "independently from the claim and agent text only."
        )
    user = {
        "claim": item["claim_description"],
        "agent_text": item["agent_text"],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=True, sort_keys=True)},
    ]


def _parse_grounding_judge_response(text: str) -> dict[str, Any]:
    payload = json.loads(_extract_json_object(text))
    if not isinstance(payload, dict):
        raise ValueError("grounding judge response JSON must be an object")
    verdict = payload.get("verdict")
    if verdict not in GROUNDING_VERDICTS:
        raise ValueError(
            f"grounding judge verdict must be one of {GROUNDING_VERDICTS}, got {verdict!r}"
        )
    rationale = payload.get("rationale")
    return {
        "verdict": verdict,
        "rationale": str(rationale) if isinstance(rationale, str) else "",
    }


def _grounding_rater_ids(specs: list[ModelJudgeSpec]) -> list[str]:
    counts: dict[str, int] = {}
    rater_ids = []
    for spec in specs:
        base = _grounding_rater_id(spec)
        counts[base] = counts.get(base, 0) + 1
        rater_ids.append(base if counts[base] == 1 else f"{base}-{counts[base]}")
    return rater_ids


def _grounding_rater_id(
    spec: ModelJudgeSpec | None,
    *,
    prefix: str = "grounding-judge",
) -> str:
    if spec is None:
        return prefix
    if spec.rater_id:
        return spec.rater_id
    value = f"{prefix}-{spec.provider}-{spec.model_id}"
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()
    return value or prefix
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_semantic_grounding.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/benchmark/semantic_grounding.py tests/unit/test_semantic_grounding.py
git commit -m "Add narrow LLM-judge calls for semantic grounding claims"
```

---

### Task 3: Aggregation and validation

**Files:**
- Modify: `src/evaluation/benchmark/semantic_grounding.py`
- Test: `tests/unit/test_semantic_grounding.py`

**Interfaces:**
- Consumes: `GROUNDING_VERDICTS`, `SATISFIED_VERDICTS`, annotation shape from Task 2. `JudgeIssue` from `judging.py`.
- Produces: `validate_grounding_annotations(annotations: list[dict]) -> list[JudgeIssue]`, `build_semantic_grounding_report(report: dict, annotations: list[dict]) -> dict`, `validate_semantic_grounding_report(report: dict) -> list[JudgeIssue]`. The built report has top-level keys `benchmark, generated_at, source_benchmark, source_benchmark_version, model_metadata, num_annotations, num_items, num_raters, overall_semantic_grounding_score, verdict_breakdown, agreement, items`, and each `items[i]` has `scenario_id, trial_index, required_score, claims` where each `claims[j]` has `scenario_id, trial_index, claim_id, verdict, num_raters`. Task 4 depends on exactly these keys.

- [ ] **Step 1: Write the failing tests**

Update the `semantic_grounding` import in `tests/unit/test_semantic_grounding.py` to:

```python
from src.evaluation.benchmark.semantic_grounding import (
    build_semantic_grounding_report,
    generate_semantic_grounding_annotations,
    iter_blinded_grounding_items,
    validate_semantic_grounding_report,
)
```

Append:

```python
def test_build_semantic_grounding_report_scores_honest_alternative_as_satisfied():
    scenario = _grounding_scenario()
    report = OpenVoiceCSBench(scenarios=[scenario]).score_agent(
        _agent_with_text(
            "I'm sorry, I can't waive that fee myself, so I've escalated it to billing."
        ),
        trials=1,
    )

    def caller(spec, messages, max_output_tokens, temperature, timeout_seconds):
        del spec, messages, max_output_tokens, temperature, timeout_seconds
        return json.dumps({"verdict": "honest_alternative"})

    annotations = generate_semantic_grounding_annotations(
        report,
        [scenario],
        judge_specs=[
            ModelJudgeSpec(provider="openrouter", model_id="judge-a"),
            ModelJudgeSpec(provider="openrouter", model_id="judge-b"),
        ],
        caller=caller,
    )
    grounding_report = build_semantic_grounding_report(report, annotations)

    assert validate_semantic_grounding_report(grounding_report) == []
    assert grounding_report["benchmark"] == "OpenVoiceCS-Bench Semantic Grounding Report"
    assert grounding_report["overall_semantic_grounding_score"] == 1.0
    assert grounding_report["verdict_breakdown"]["honest_alternative"] == 1
    assert grounding_report["verdict_breakdown"]["not_grounded"] == 0
    assert len(grounding_report["items"]) == 1
    assert grounding_report["items"][0]["required_score"] == 1.0


def test_build_semantic_grounding_report_scores_not_grounded_as_missing():
    scenario = _grounding_scenario()
    report = OpenVoiceCSBench(scenarios=[scenario]).score_agent(
        _agent_with_text("Sure, I've updated your account."), trials=1
    )

    def caller(spec, messages, max_output_tokens, temperature, timeout_seconds):
        del spec, messages, max_output_tokens, temperature, timeout_seconds
        return json.dumps({"verdict": "not_grounded", "rationale": "no mention of the fee"})

    annotations = generate_semantic_grounding_annotations(
        report,
        [scenario],
        judge_specs=[
            ModelJudgeSpec(provider="openrouter", model_id="judge-a"),
            ModelJudgeSpec(provider="openrouter", model_id="judge-b"),
        ],
        caller=caller,
    )
    grounding_report = build_semantic_grounding_report(report, annotations)

    assert grounding_report["overall_semantic_grounding_score"] == 0.0
    assert grounding_report["items"][0]["required_score"] == 0.0


def test_build_semantic_grounding_report_uses_adjudicator_to_break_ties():
    scenario = _grounding_scenario()
    report = OpenVoiceCSBench(scenarios=[scenario]).score_agent(
        _agent_with_text("I've escalated this."), trials=1
    )

    def caller(spec, messages, max_output_tokens, temperature, timeout_seconds):
        del messages, max_output_tokens, temperature, timeout_seconds
        verdict = {"judge-a": "grounded", "judge-b": "not_grounded"}.get(
            spec.model_id, "honest_alternative"
        )
        return json.dumps({"verdict": verdict})

    annotations = generate_semantic_grounding_annotations(
        report,
        [scenario],
        judge_specs=[
            ModelJudgeSpec(provider="openrouter", model_id="judge-a"),
            ModelJudgeSpec(provider="openrouter", model_id="judge-b"),
        ],
        adjudicator=ModelJudgeSpec(provider="openrouter", model_id="judge-c"),
        caller=caller,
    )
    grounding_report = build_semantic_grounding_report(report, annotations)

    assert grounding_report["items"][0]["claims"][0]["verdict"] == "honest_alternative"
    assert grounding_report["items"][0]["required_score"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_semantic_grounding.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_semantic_grounding_report'`

- [ ] **Step 3: Write the implementation**

Replace the top of the file, from `from __future__ import annotations` down
through the closing `)` of the `judging` import, with:

```python
from __future__ import annotations

import json
import re
import statistics
import time
from typing import Any

from src.evaluation.benchmark.judging import (
    JudgeIssue,
    ModelJudgeCaller,
    ModelJudgeSpec,
    _extract_json_object,
    _trial_index_from_report,
    call_openai_compatible_model_judge,
)
```

Add at the end of the file:

```python
def validate_grounding_annotations(annotations: list[dict[str, Any]]) -> list[JudgeIssue]:
    """Return structural issues in a list of semantic-grounding annotations."""
    issues: list[JudgeIssue] = []
    if not isinstance(annotations, list):
        return [JudgeIssue("<annotations>", "<root>", "must be a list")]
    for index, annotation in enumerate(annotations):
        path = f"annotations[{index}]"
        if not isinstance(annotation, dict):
            issues.append(JudgeIssue("<annotations>", path, "must be an object"))
            continue
        item_id = str(annotation.get("item_id") or f"<item-{index}>")
        for field in ("item_id", "scenario_id", "trial_index", "claim_id", "rater_id", "verdict"):
            if field not in annotation:
                issues.append(JudgeIssue(item_id, f"{path}.{field}", "missing required field"))
        if "verdict" in annotation and annotation["verdict"] not in GROUNDING_VERDICTS:
            issues.append(
                JudgeIssue(item_id, f"{path}.verdict", f"must be one of {GROUNDING_VERDICTS}")
            )
        trial_index = annotation.get("trial_index")
        if "trial_index" in annotation and (
            isinstance(trial_index, bool) or not isinstance(trial_index, int)
        ):
            issues.append(JudgeIssue(item_id, f"{path}.trial_index", "must be an integer"))
    return issues


def build_semantic_grounding_report(
    report: dict[str, Any],
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate semantic-grounding annotations into a report-shaped summary."""
    issues = validate_grounding_annotations(annotations)
    if issues:
        formatted = "\n".join(
            f"- {issue.item_id}::{issue.path}: {issue.message}" for issue in issues
        )
        raise ValueError(
            f"OpenVoiceCS semantic grounding annotation validation failed:\n{formatted}"
        )

    by_claim: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for annotation in annotations:
        key = (annotation["scenario_id"], annotation["trial_index"], annotation["claim_id"])
        by_claim.setdefault(key, []).append(annotation)

    claim_results = []
    verdict_breakdown = {verdict: 0 for verdict in GROUNDING_VERDICTS}
    for (scenario_id, trial_index, claim_id), claim_annotations in sorted(by_claim.items()):
        verdict = _majority_verdict(claim_annotations)
        verdict_breakdown[verdict] += 1
        claim_results.append({
            "scenario_id": scenario_id,
            "trial_index": trial_index,
            "claim_id": claim_id,
            "verdict": verdict,
            "num_raters": len({a["rater_id"] for a in claim_annotations}),
        })

    by_trial: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for claim_result in claim_results:
        key = (claim_result["scenario_id"], claim_result["trial_index"])
        by_trial.setdefault(key, []).append(claim_result)

    trial_items = []
    for (scenario_id, trial_index), claims in sorted(by_trial.items()):
        satisfied = sum(1 for claim in claims if claim["verdict"] in SATISFIED_VERDICTS)
        trial_items.append({
            "scenario_id": scenario_id,
            "trial_index": trial_index,
            "required_score": round(satisfied / len(claims), 6),
            "claims": claims,
        })

    return {
        "benchmark": "OpenVoiceCS-Bench Semantic Grounding Report",
        "generated_at": time.strftime("%Y-%m-%d"),
        "source_benchmark": report.get("benchmark"),
        "source_benchmark_version": report.get("benchmark_version"),
        "model_metadata": report.get("model_metadata", {}),
        "num_annotations": len(annotations),
        "num_items": len(claim_results),
        "num_raters": len({a["rater_id"] for a in annotations}),
        "overall_semantic_grounding_score": round(
            statistics.mean([item["required_score"] for item in trial_items])
            if trial_items else 0.0,
            6,
        ),
        "verdict_breakdown": verdict_breakdown,
        "agreement": _grounding_agreement(by_claim),
        "items": trial_items,
    }


def _majority_verdict(annotations: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for annotation in annotations:
        counts[annotation["verdict"]] = counts.get(annotation["verdict"], 0) + 1
    adjudicator_verdict = next(
        (a["verdict"] for a in annotations if a.get("judge", {}).get("adjudicator")),
        None,
    )
    best_count = max(counts.values())
    leaders = [verdict for verdict, count in counts.items() if count == best_count]
    if len(leaders) == 1:
        return leaders[0]
    if adjudicator_verdict is not None:
        return adjudicator_verdict
    return "not_grounded"


def _grounding_agreement(
    by_claim: dict[tuple[str, int, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    multi_rater = [
        claims for claims in by_claim.values()
        if len({a["rater_id"] for a in claims}) >= 2
    ]
    if not multi_rater:
        return {"num_multi_rater_items": 0, "exact_agreement_rate": None}
    agreeing = sum(
        1 for claims in multi_rater
        if len({
            a["verdict"] for a in claims if not a.get("judge", {}).get("adjudicator")
        }) <= 1
    )
    return {
        "num_multi_rater_items": len(multi_rater),
        "exact_agreement_rate": round(agreeing / len(multi_rater), 4),
    }


def validate_semantic_grounding_report(report: dict[str, Any]) -> list[JudgeIssue]:
    """Validate an aggregated semantic-grounding report's structure and ranges."""
    issues: list[JudgeIssue] = []
    if not isinstance(report, dict):
        return [JudgeIssue("<grounding-report>", "<root>", "must be an object")]

    required = {
        "benchmark", "generated_at", "num_annotations", "num_items", "num_raters",
        "overall_semantic_grounding_score", "verdict_breakdown", "agreement", "items",
    }
    for field in sorted(required - set(report)):
        issues.append(JudgeIssue("<grounding-report>", field, "missing required field"))
    if issues:
        return issues

    if report.get("benchmark") != "OpenVoiceCS-Bench Semantic Grounding Report":
        issues.append(
            JudgeIssue(
                "<grounding-report>",
                "benchmark",
                "must be OpenVoiceCS-Bench Semantic Grounding Report",
            )
        )

    score = report.get("overall_semantic_grounding_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not (0.0 <= float(score) <= 1.0):
        issues.append(
            JudgeIssue("<grounding-report>", "overall_semantic_grounding_score", "must be between 0 and 1")
        )

    items = report.get("items")
    if not isinstance(items, list):
        issues.append(JudgeIssue("<grounding-report>", "items", "must be a list"))
        items = []
    for index, item in enumerate(items):
        path = f"items[{index}]"
        if not isinstance(item, dict):
            issues.append(JudgeIssue("<grounding-report>", path, "must be an object"))
            continue
        scenario_id = str(item.get("scenario_id") or f"<item-{index}>")
        required_score = item.get("required_score")
        if (
            isinstance(required_score, bool)
            or not isinstance(required_score, (int, float))
            or not (0.0 <= float(required_score) <= 1.0)
        ):
            issues.append(JudgeIssue(scenario_id, f"{path}.required_score", "must be between 0 and 1"))
        claims = item.get("claims")
        if not isinstance(claims, list) or not claims:
            issues.append(JudgeIssue(scenario_id, f"{path}.claims", "must be a non-empty list"))
    return issues
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_semantic_grounding.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/benchmark/semantic_grounding.py tests/unit/test_semantic_grounding.py
git commit -m "Add semantic grounding report aggregation and validation"
```

---

### Task 4: Additive merge into benchmark reports

**Files:**
- Modify: `src/evaluation/benchmark/semantic_grounding.py`
- Test: `tests/unit/test_semantic_grounding.py`

**Interfaces:**
- Consumes: `build_semantic_grounding_report` output shape from Task 3.
- Produces: `apply_semantic_grounding_report(report: dict, semantic_report: dict) -> dict`. Returns a deep copy of `report` with `report["semantic_grounding"] = {"score": float | None, "coverage": float, "num_judged_trials": int, "verdict_breakdown": dict, "source_report": dict}` and, per matched trial, `trial["semantic_grounding_check"] = {"score": float, "claims": list[dict], "source": "audited_grounding_judge"}`. Does not modify `metric_scores`, `overall_score`, or any existing trial key.

- [ ] **Step 1: Write the failing test**

Update the `semantic_grounding` import block to add `apply_semantic_grounding_report`:

```python
from src.evaluation.benchmark.semantic_grounding import (
    apply_semantic_grounding_report,
    build_semantic_grounding_report,
    generate_semantic_grounding_annotations,
    iter_blinded_grounding_items,
    validate_semantic_grounding_report,
)
```

Append:

```python
def test_apply_semantic_grounding_report_is_additive():
    scenario = _grounding_scenario()
    report = OpenVoiceCSBench(scenarios=[scenario]).score_agent(
        _agent_with_text(
            "I'm sorry, I can't waive that fee myself, so I've escalated it to billing."
        ),
        trials=1,
    )
    original = deepcopy(report)

    def caller(spec, messages, max_output_tokens, temperature, timeout_seconds):
        del spec, messages, max_output_tokens, temperature, timeout_seconds
        return json.dumps({"verdict": "honest_alternative"})

    annotations = generate_semantic_grounding_annotations(
        report,
        [scenario],
        judge_specs=[
            ModelJudgeSpec(provider="openrouter", model_id="judge-a"),
            ModelJudgeSpec(provider="openrouter", model_id="judge-b"),
        ],
        caller=caller,
    )
    grounding_report = build_semantic_grounding_report(report, annotations)
    graded = apply_semantic_grounding_report(report, grounding_report)

    # Literal factual_grounding is 0.0 here -- the honest-failure text matches
    # none of the required claim's `any_terms` -- proving the semantic path
    # disagrees with (and does not touch) the literal one.
    assert original["metric_scores"]["factual_grounding"] == 0.0
    assert graded["metric_scores"] == original["metric_scores"]
    assert graded["overall_score"] == original["overall_score"]
    assert (
        graded["results"][0]["trials"][0]["grounding_check"]
        == original["results"][0]["trials"][0]["grounding_check"]
    )

    assert graded["semantic_grounding"]["score"] == 1.0
    assert graded["semantic_grounding"]["num_judged_trials"] == 1
    assert graded["semantic_grounding"]["coverage"] == 1.0
    trial_check = graded["results"][0]["trials"][0]["semantic_grounding_check"]
    assert trial_check["score"] == 1.0
    assert trial_check["source"] == "audited_grounding_judge"
    assert validate_report(graded) == []


def test_apply_semantic_grounding_report_leaves_unmatched_trials_alone():
    scenario = _grounding_scenario()
    other = _grounding_scenario()
    other["id"] = "other-001"
    del other["oracle"]["grounding"]
    report = OpenVoiceCSBench(scenarios=[scenario, other]).score_agent(
        _agent_with_text("Escalated."), trials=1
    )

    def caller(spec, messages, max_output_tokens, temperature, timeout_seconds):
        del spec, messages, max_output_tokens, temperature, timeout_seconds
        return json.dumps({"verdict": "honest_alternative"})

    annotations = generate_semantic_grounding_annotations(
        report,
        [scenario, other],
        judge_specs=[ModelJudgeSpec(provider="openrouter", model_id="judge-a")],
        caller=caller,
    )
    grounding_report = build_semantic_grounding_report(report, annotations)
    graded = apply_semantic_grounding_report(report, grounding_report)

    assert "semantic_grounding_check" not in graded["results"][1]["trials"][0]
    assert graded["semantic_grounding"]["coverage"] == 0.5
    assert graded["semantic_grounding"]["num_judged_trials"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_semantic_grounding.py -v`
Expected: FAIL — `ImportError: cannot import name 'apply_semantic_grounding_report'`

- [ ] **Step 3: Write the implementation**

Replace the top of the file, from `from __future__ import annotations` down
through the closing `)` of the `judging` import, with:

```python
from __future__ import annotations

import json
import re
import statistics
import time
from copy import deepcopy
from typing import Any

from src.evaluation.benchmark.judging import (
    JudgeIssue,
    ModelJudgeCaller,
    ModelJudgeSpec,
    _extract_json_object,
    _trial_index_from_report,
    call_openai_compatible_model_judge,
)
```

Add at the end of the file:

```python
def apply_semantic_grounding_report(
    report: dict[str, Any],
    semantic_report: dict[str, Any],
) -> dict[str, Any]:
    """Attach an aggregated semantic grounding report to an OpenVoiceCS report.

    Additive only: never modifies ``metric_scores``, ``overall_score``, or
    any existing ``trial["grounding_check"]`` entry produced by the literal
    matcher.
    """
    updated = deepcopy(report)
    items_by_key = {
        (str(item.get("scenario_id")), int(item.get("trial_index", 0))): item
        for item in semantic_report.get("items", [])
        if isinstance(item, dict)
    }
    assigned_scores = []
    assigned_count = 0
    total_trials = 0
    for result in updated.get("results", []):
        scenario_id = str(result.get("id"))
        for trial in result.get("trials", []):
            total_trials += 1
            trial_index = trial.get("trial_index", 0)
            item = items_by_key.get((scenario_id, trial_index))
            if item is None:
                continue
            trial["semantic_grounding_check"] = {
                "score": item["required_score"],
                "claims": item["claims"],
                "source": "audited_grounding_judge",
            }
            assigned_scores.append(item["required_score"])
            assigned_count += 1

    updated["semantic_grounding"] = {
        "score": round(statistics.mean(assigned_scores), 6) if assigned_scores else None,
        "coverage": round(assigned_count / total_trials, 4) if total_trials else 0.0,
        "num_judged_trials": assigned_count,
        "verdict_breakdown": semantic_report.get("verdict_breakdown", {}),
        "source_report": {
            "num_annotations": semantic_report.get("num_annotations"),
            "num_raters": semantic_report.get("num_raters"),
            "agreement": semantic_report.get("agreement"),
        },
    }
    return updated
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_semantic_grounding.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/benchmark/semantic_grounding.py tests/unit/test_semantic_grounding.py
git commit -m "Add additive merge of semantic grounding into benchmark reports"
```

---

### Task 5: CLI subcommands

**Files:**
- Modify: `scripts/run_openvoicecs.py`

**Interfaces:**
- Consumes: `generate_semantic_grounding_annotations`, `build_semantic_grounding_report`, `apply_semantic_grounding_report`, `validate_semantic_grounding_report` (Tasks 2-4). Already-imported `parse_model_judge_spec`, `write_judge_annotations_jsonl`, `load_workspace_env`, `validate_report`, `DEFAULT_SCENARIO_PATH`, `OpenVoiceCSBench`.
- Produces: `semantic-grounding` and `apply-semantic-grounding-report` CLI subcommands.

There is no dedicated unit test for CLI wiring in this codebase (`model-judge` has none either — it's exercised through the library-level tests in Task 1-4 plus manual/integration use). Verify this task with the smoke commands in Step 4 instead of pytest.

- [ ] **Step 1: Add the import**

In `scripts/run_openvoicecs.py`, find this existing import block (currently ends around line 79):

```python
from src.evaluation.benchmark.judging import (
    DEFAULT_JUDGE_ANNOTATION_PACKAGE_PATH,
    DEFAULT_JUDGE_PROTOCOL_PATH,
    DEFAULT_JUDGE_RUBRIC_PATH,
    DEFAULT_JUDGE_STUDY_PATH,
    apply_judge_report,
    apply_judge_report_from_files,
    build_judge_report,
    build_judge_report_from_files,
    generate_model_judge_annotations,
    load_judge_protocol,
    parse_model_judge_spec,
    validate_judge_annotation_package_file,
    validate_judge_protocol_file,
    validate_judge_report_file,
    validate_judge_rubric_file,
    validate_judge_study_manifest_file,
    write_judge_annotations_jsonl,
)
```

Immediately after it (before `from src.evaluation.benchmark.openvoicecs import (`), add:

```python
from src.evaluation.benchmark.semantic_grounding import (
    apply_semantic_grounding_report,
    build_semantic_grounding_report,
    generate_semantic_grounding_annotations,
    validate_semantic_grounding_report,
)
```

- [ ] **Step 2: Add the `cmd_semantic_grounding` and `cmd_apply_semantic_grounding_report` functions**

Find `def cmd_compare(args: argparse.Namespace) -> None:` (currently around line 859, right after `cmd_model_judge`). Insert the two new functions immediately before it:

```python
def cmd_semantic_grounding(args: argparse.Namespace) -> None:
    load_workspace_env(args.env)
    with open(args.report, encoding="utf-8") as f:
        source_report = json.load(f)
    scenarios = OpenVoiceCSBench.load(args.scenarios).scenarios
    judge_specs = [parse_model_judge_spec(value) for value in args.judge]
    if len(judge_specs) < 2:
        print(
            "semantic-grounding requires at least two --judge specs for audited judging",
            file=sys.stderr,
        )
        raise SystemExit(2)
    adjudicator = parse_model_judge_spec(args.adjudicator) if args.adjudicator else None

    annotations = generate_semantic_grounding_annotations(
        source_report,
        scenarios,
        judge_specs=judge_specs,
        adjudicator=adjudicator,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        timeout_seconds=args.timeout_seconds,
    )
    annotations_output = Path(args.annotations_output)
    write_judge_annotations_jsonl(annotations, annotations_output)

    grounding_report = build_semantic_grounding_report(source_report, annotations)
    issues = validate_semantic_grounding_report(grounding_report)
    if issues:
        print("Semantic grounding report validation failed:")
        for issue in issues:
            print(f"  {issue.item_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)
    grounding_report_output = Path(args.grounding_report_output)
    grounding_report_output.parent.mkdir(parents=True, exist_ok=True)
    with open(grounding_report_output, "w", encoding="utf-8") as f:
        json.dump(grounding_report, f, indent=2)

    graded_report = None
    graded_report_output = Path(args.graded_report_output) if args.graded_report_output else None
    if graded_report_output:
        graded_report = apply_semantic_grounding_report(source_report, grounding_report)
        issues = validate_report(graded_report)
        if issues:
            print("Graded report validation failed:")
            for issue in issues:
                print(f"  {issue.scenario_id}::{issue.path}: {issue.message}")
            raise SystemExit(1)
        graded_report_output.parent.mkdir(parents=True, exist_ok=True)
        with open(graded_report_output, "w", encoding="utf-8") as f:
            json.dump(graded_report, f, indent=2)

    _print_semantic_grounding_result(
        annotations=annotations,
        annotations_output=annotations_output,
        grounding_report=grounding_report,
        grounding_report_output=grounding_report_output,
        graded_report_output=graded_report_output,
    )


def cmd_apply_semantic_grounding_report(args: argparse.Namespace) -> None:
    with open(args.report, encoding="utf-8") as f:
        source_report = json.load(f)
    with open(args.grounding_report, encoding="utf-8") as f:
        grounding_report = json.load(f)
    issues = validate_semantic_grounding_report(grounding_report)
    if issues:
        print("Semantic grounding report validation failed:")
        for issue in issues:
            print(f"  {issue.item_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)

    graded_report = apply_semantic_grounding_report(source_report, grounding_report)
    issues = validate_report(graded_report)
    if issues:
        print("Graded report validation failed:")
        for issue in issues:
            print(f"  {issue.scenario_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)

    semantic = graded_report["semantic_grounding"]
    print(f"Semantic grounding score: {semantic['score']}")
    print(f"Coverage: {semantic['coverage']:.1%}")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(graded_report, f, indent=2)
        print(f"\nSaved graded report to {output}")
```

- [ ] **Step 3: Add the print helper and argparse subparsers**

Find `def _print_model_judge_result(` (around line 1855) and locate its closing `)` / end of function body (it ends right before the next `def`). Add this new function immediately after `_print_model_judge_result` finishes:

```python
def _print_semantic_grounding_result(
    *,
    annotations: list[dict[str, Any]],
    annotations_output: Path,
    grounding_report: dict[str, Any],
    grounding_report_output: Path,
    graded_report_output: Path | None,
) -> None:
    print("\nOPENVOICECS-BENCH SEMANTIC GROUNDING")
    print("=" * 88)
    print(f"Annotations:            {len(annotations)}")
    print(f"Items (trial x claim):  {grounding_report.get('num_items', 0)}")
    print(f"Raters:                 {grounding_report.get('num_raters', 0)}")
    print(f"Semantic score:         {grounding_report.get('overall_semantic_grounding_score', 0):.1%}")
    breakdown = grounding_report.get("verdict_breakdown", {})
    print(
        "Verdicts:               "
        f"grounded={breakdown.get('grounded', 0)} "
        f"honest_alternative={breakdown.get('honest_alternative', 0)} "
        f"not_grounded={breakdown.get('not_grounded', 0)}"
    )
    print(f"Annotations saved to:   {annotations_output}")
    print(f"Grounding report saved to: {grounding_report_output}")
    if graded_report_output:
        print(f"Graded report saved to:    {graded_report_output}")
```

Now find the `model_judge.set_defaults(func=cmd_model_judge)` line followed by the `apply_judge = subparsers.add_parser(...)` block and its `apply_judge.set_defaults(func=cmd_apply_judge_report)` line (this is the block right before `compare = subparsers.add_parser(...)`, around line 2444-2452). Insert the following immediately after `apply_judge.set_defaults(func=cmd_apply_judge_report)` and before `compare = subparsers.add_parser(`:

```python
    semantic_grounding = subparsers.add_parser(
        "semantic-grounding",
        help="Call audited grounding judges to re-score required-claim grounding semantically",
    )
    semantic_grounding.add_argument("report", help="Source OpenVoiceCS report JSON")
    semantic_grounding.add_argument(
        "--judge",
        action="append",
        required=True,
        help="Judge spec as provider:model_id, repeat for two or more judges",
    )
    semantic_grounding.add_argument(
        "--adjudicator",
        default=None,
        help="Optional tie-breaker judge spec as provider:model_id",
    )
    semantic_grounding.add_argument("--scenarios", default=str(DEFAULT_SCENARIO_PATH))
    semantic_grounding.add_argument("--annotations-output", required=True)
    semantic_grounding.add_argument("--grounding-report-output", required=True)
    semantic_grounding.add_argument("--graded-report-output", default=None)
    semantic_grounding.add_argument("--max-output-tokens", type=int, default=200)
    semantic_grounding.add_argument("--temperature", type=float, default=0.0)
    semantic_grounding.add_argument("--timeout-seconds", type=float, default=60.0)
    semantic_grounding.add_argument(
        "--env",
        default=".env",
        help="Environment file with provider API keys",
    )
    semantic_grounding.set_defaults(func=cmd_semantic_grounding)

    apply_semantic_grounding = subparsers.add_parser(
        "apply-semantic-grounding-report",
        help="Attach an aggregated semantic grounding report to a benchmark report",
    )
    apply_semantic_grounding.add_argument("report", help="Source OpenVoiceCS report JSON")
    apply_semantic_grounding.add_argument(
        "grounding_report", help="Aggregated semantic grounding report JSON"
    )
    apply_semantic_grounding.add_argument("--output", default=None)
    apply_semantic_grounding.set_defaults(func=cmd_apply_semantic_grounding_report)
```

- [ ] **Step 4: Smoke-test the CLI end to end with a fake judge**

There's no built-in way to inject a fake `caller` from the command line (by design — the CLI always uses the real HTTP caller), so smoke-test the parser wiring and the library path together with a tiny throwaway Python script instead of a live API call:

```bash
python -m pytest tests/unit/test_semantic_grounding.py -v
python -c "
import subprocess, sys
result = subprocess.run(
    [sys.executable, 'scripts/run_openvoicecs.py', '--help'],
    capture_output=True, text=True, check=True,
)
assert 'semantic-grounding' in result.stdout, result.stdout
assert 'apply-semantic-grounding-report' in result.stdout, result.stdout
print('CLI subcommands registered OK')
"
python -c "
import subprocess, sys
result = subprocess.run(
    [sys.executable, 'scripts/run_openvoicecs.py', 'semantic-grounding', '--help'],
    capture_output=True, text=True, check=True,
)
assert '--judge' in result.stdout
assert '--adjudicator' in result.stdout
assert '--scenarios' in result.stdout
print('semantic-grounding flags OK')
"
```

Expected: all three commands print their success line and exit 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_openvoicecs.py
git commit -m "Wire semantic-grounding CLI subcommands"
```

---

### Task 6: Documentation

**Files:**
- Modify: `docs/known-limitations.md`

- [ ] **Step 1: Add the section 7 addendum**

Find this paragraph in `docs/known-limitations.md` (end of section 7):

```
Scores span 0.047–0.323 across the ranked cohort at weight 0.20, enough to
reorder the top of the leaderboard. Treat ranks 1–2 as tied. This is the
strongest remaining argument for a semantic grader and the most valuable
contribution anyone can make.
```

Replace it with (same text, plus a new paragraph after it):

```
Scores span 0.047–0.323 across the ranked cohort at weight 0.20, enough to
reorder the top of the leaderboard. Treat ranks 1–2 as tied. This is the
strongest remaining argument for a semantic grader and the most valuable
contribution anyone can make.

**Partly addressed.** An opt-in semantic grader is now available:
`scripts/run_openvoicecs.py semantic-grounding` /
`apply-semantic-grounding-report`, following the same blinded, audited
pattern as the model-judge subjective-quality pipeline (section 13). It
classifies each required claim as `grounded`, `honest_alternative` (the
agent didn't make the claim but truthfully reported a different outcome —
this is what fixes the honest-failure-report bullet above), or
`not_grounded` via a narrow LLM-judge call, and merges into a report's
`semantic_grounding` field (`src/evaluation/benchmark/semantic_grounding.py`)
without altering `factual_grounding` or `overall_score` — the literal
matcher above remains the default and only score that feeds the leaderboard.
Judge-to-human agreement on this claim-level classification has not been
measured yet; treat it as provisional, same caveat as section 13.
```

- [ ] **Step 2: Verify the doc still reads correctly**

Run: `python -c "import pathlib; text = pathlib.Path('docs/known-limitations.md').read_text(encoding='utf-8'); assert 'Partly addressed' in text; assert text.count('## Open limitations') == 1; print('doc OK')"`
Expected: prints `doc OK`

- [ ] **Step 3: Commit**

```bash
git add docs/known-limitations.md
git commit -m "Document opt-in semantic grounding check in known-limitations"
```

---

### Task 7: Full verification pass

**Files:** none (verification only)

- [ ] **Step 1: Run the new test module in isolation**

Run: `python -m pytest tests/unit/test_semantic_grounding.py -v`
Expected: all tests PASS

- [ ] **Step 2: Lint the touched files**

Run: `python -m ruff check src/evaluation/benchmark/semantic_grounding.py scripts/run_openvoicecs.py tests/unit/test_semantic_grounding.py`
Expected: no new findings (this repo carries pre-existing `E501` debt elsewhere, but these are new/lightly-touched files — if ruff reports anything in the new module, fix it before moving on; `scripts/run_openvoicecs.py` may report pre-existing unrelated `E501`s — ignore those, fix only lines you added)

- [ ] **Step 3: Run the full local gate**

Run: `make check`
Expected: PASS. This runs scenario validation, review-manifest validation, submission-intake validation, the strict release gate, release-bundle verification, and the full unit test suite — confirms the new module didn't break anything existing, and in particular that `factual_grounding` / `overall_score` are bit-for-bit unchanged across the whole suite (Task 4's regression test covers one scenario directly; this covers the full corpus indirectly by re-running every existing test).

- [ ] **Step 4: Run the full test suite explicitly**

Run: `python -m pytest tests/unit -v`
Expected: all PASS, including the pre-existing `tests/unit/test_openvoicecs.py`, `tests/unit/test_judging.py`, and `tests/unit/test_scoring_validity.py` — none of these should change behavior.

No commit for this task — it's verification of work already committed in Tasks 1-6. If any step fails, fix the root cause in the relevant earlier task's files and re-run this task's steps before considering the plan complete.
