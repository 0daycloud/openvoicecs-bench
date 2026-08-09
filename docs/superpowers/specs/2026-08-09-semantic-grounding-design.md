# Semantic factual-grounding check (design)

Status: approved for implementation.
Addresses: `docs/known-limitations.md` section 7, priority-1 item in "Priority order".

## Problem

`check_factual_grounding` in `src/evaluation/benchmark/openvoicecs.py` matches
`oracle.grounding.required_claims[].any_terms` as literal regex/substring
patterns against the agent's stated text (`_matches_any`). This conflates
three distinct things under one `factual_grounding` score (weight 0.20):

1. Synonymy misses — a claim is true and stated, just not in the pinned
   wording (`"no change fee"` vs. an agent saying `"rebooked at no charge"`).
2. Genuine omissions — correctly caught today, must stay caught.
3. Honest failure reports — the agent truthfully says it couldn't complete an
   action and escalated, but the required claim assumed success, so the
   agent is penalized for accuracy.

Removing `factual_grounding` from the aggregate reshuffles 36 of 44 ranked
models — it is the largest source of mid-table noise in the v0.2 sweep.

## Non-goals

- Not touching `check_factual_grounding`, `metric_scores.factual_grounding`,
  or `overall_score`. The literal matcher stays the default and only scoring
  path unless a maintainer explicitly runs the new opt-in pipeline and merges
  it in — this is what makes the change additive rather than
  scoring-affecting-by-default (CONTRIBUTING.md requires PRs to declare
  "whether scoring behavior changed"; the answer here is no, by default).
- Not touching `forbidden_claims` / hallucination detection. Those already
  use regex patterns (more flexible than required-claim substring matching)
  and are not what section 7 complains about.
- Not adding new scenario schema fields or editing scenario JSON. The judge
  reads `oracle.grounding.required_claims[].id` and `.any_terms`, both of
  which already exist on every scenario. No `data/openvoicecs/scenarios_*`
  change, so none of CONTRIBUTING's "Scenario Changes" process (splits,
  provenance, changelog, scenario review) applies.
- Embedding-similarity backend: not in this PR. The interface is left open
  for it (see "Extensibility"), but a pure similarity score can't distinguish
  "claim absent" from "claim honestly contradicted by a truthful failure
  report" — the single biggest complaint in section 7 — so it wouldn't fix
  the part of the bug that matters most. LLM-judge does, in one call.

## Architecture

New module `src/evaluation/benchmark/semantic_grounding.py`, structured like
the existing `judging.py` model-judge pipeline (same file, adjacent
concerns: both are "needs an LLM call, must stay blinded and auditable").
Offline and post-hoc, exactly like model-judge:

```
score_agent() report (unchanged)
        |
        v
generate_semantic_grounding_annotations(report, judge_specs, ...)
        |  (one call per required claim per trial, blinded)
        v
build_semantic_grounding_report(report, annotations)
        |  (aggregate, validate)
        v
apply_semantic_grounding_report(report, semantic_report)
        |  (additive merge)
        v
report["semantic_grounding"] = {...}   # new top-level field
report["results"][i]["trials"][j]["semantic_grounding_check"] = {...}
```

`metric_scores`, `overall_score`, and the existing `trial["grounding_check"]`
(from `check_factual_grounding`) are untouched by `apply_semantic_grounding_report`.
This mirrors `apply_judge_report`, which writes `conversation_experience` /
`conversation_experience_score` alongside the deterministic metrics without
altering them.

Reused from `judging.py` (imported, not duplicated): `ModelJudgeSpec`,
`ModelJudgeCaller`, `call_openai_compatible_model_judge`, `parse_model_judge_spec`,
and the private utilities `_extract_json_object` and `_trial_index_from_report`
(pure JSON/formatting helpers with no judging-specific coupling — importing
them avoids re-deriving the same JSON-extraction and trial-index-normalization
logic a second time).

## Data flow in detail

### 1. Blinded items

`report["results"][i]` (as produced by `_aggregate_scenario_trials`) does not
carry `oracle.grounding.required_claims` — only `trial["grounding_check"]`'s
`missing_required_claims` (the ones the literal matcher missed) is present,
which isn't the full claim set an independent judge needs to re-evaluate.
So this function takes the original scenario suite alongside the report:

```python
def iter_blinded_grounding_items(
    report: dict[str, Any],
    scenarios: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One item per (trial, required claim). Empty required_claims -> no items.

    `scenarios` is the same suite the report was produced from (e.g.
    `OpenVoiceCSBench.load(path).scenarios`) -- matched to report results by
    `id`, which is stable for audio variants too (`build_audio_variant_scenarios`
    copies the base scenario's oracle under the variant's own id).
    """
```

Per item:

```json
{
  "item_id": "<scenario_id>:<trial_index>:<claim_id>",
  "scenario_id": "...",
  "trial_index": 0,
  "claim_id": "fee_waived",
  "claim_description": "no change fee / no fee / fee waiver",
  "agent_text": "...agent's stated text, joined agent turns..."
}
```

`claim_description` is `" / ".join(claim["any_terms"])` — the same
information the literal matcher already uses to match, just handed to the
judge as context for what the claim *means* rather than as a pattern to
string-match. No scenario identity, model identity, oracle pass/fail, tool
calls, or other metrics are included — matches the blinding level of
`iter_blinded_judge_items`.

Scenarios with no `oracle.grounding.required_claims` produce no items and
are silently skipped (nothing to judge).

### 2. Narrow judge call

```python
def generate_semantic_grounding_annotations(
    report: dict[str, Any],
    scenarios: list[dict[str, Any]],
    *,
    judge_specs: list[ModelJudgeSpec],
    adjudicator: ModelJudgeSpec | None = None,
    disagreement_threshold: float = 0.0,  # any disagreement on verdict triggers adjudication
    caller: ModelJudgeCaller | None = None,
    max_output_tokens: int = 200,
    temperature: float = 0.0,
    timeout_seconds: float = 60.0,
) -> list[dict[str, Any]]:
```

System prompt (short, narrow — this is the "is this claim grounded in the
agent's stated text" scope from the contribution priority list, not a
general quality judge):

> You are checking whether one required claim is grounded in an AI customer
> service agent's stated reply. Given the claim and the agent's text,
> respond with exactly one verdict:
> - `grounded` — the reply conveys this claim in substance, in any wording.
> - `honest_alternative` — the reply does not make this claim, but it
>   honestly and accurately reports a different outcome (a failure,
>   inability, or escalation) rather than falsely asserting the claim.
> - `not_grounded` — the claim is neither made nor honestly addressed;
>   it is simply missing.
> Return only JSON: `{"verdict": "...", "rationale": "one sentence"}`.

One call per item (per claim, per trial) — kept narrow and cheap
(`max_output_tokens=200`) rather than batching all claims for a trial into
one call, so a judge failure or malformed response is isolated to a single
claim instead of invalidating a whole trial's claims.

Per-item annotation shape (one per judge spec, plus adjudicator when
triggered — same pattern as `_score_blinded_item_with_model_judge`):

```json
{
  "item_id": "...",
  "scenario_id": "...",
  "claim_id": "fee_waived",
  "rater_id": "grounding-judge-openrouter-anthropic-claude-sonnet-4.6",
  "verdict": "grounded",
  "rationale": "...",
  "judge": {"type": "audited_grounding_judge", "provider": "...", "model_id": "...", "adjudicator": false}
}
```

Adjudication trigger: any disagreement between the first two raters'
verdicts (categorical, so `disagreement_threshold` is really a boolean gate
— default any mismatch adjudicates; kept as a parameter for symmetry with
model-judge and to allow a future non-strict mode).

### 3. Aggregation

```python
def build_semantic_grounding_report(
    report: dict[str, Any],
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
```

Per trial: `required_score = count(verdict in {grounded, honest_alternative}) / count(claims)`
when the trial has required claims; trials with none are excluded from the
mean (not scored as 1.0 or 0.0 — `measured: false`-style exclusion, same
principle as `measurement_coverage` elsewhere in this codebase). Majority
verdict per claim when multiple raters/adjudicator are present (ties broken
by adjudicator verdict when one ran, else `not_grounded` — the conservative
default matching the literal matcher's fail-closed behavior).

Report shape (mirrors `build_judge_report`'s envelope fields so it validates
and displays the same way):

```json
{
  "benchmark": "OpenVoiceCS-Bench Semantic Grounding Report",
  "generated_at": "2026-08-09",
  "source_benchmark": "OpenVoiceCS-Bench",
  "source_benchmark_version": "...",
  "num_annotations": 0,
  "num_items": 0,
  "num_raters": 0,
  "overall_semantic_grounding_score": 0.0,
  "verdict_breakdown": {"grounded": 0, "honest_alternative": 0, "not_grounded": 0},
  "agreement": {"...": "..."},
  "items": [
    {"scenario_id": "...", "trial_index": 0, "required_score": 1.0,
     "claims": [{"claim_id": "...", "verdict": "grounded", "num_raters": 2}]}
  ]
}
```

`validate_semantic_grounding_report(report) -> list[JudgeIssue]` mirrors
`validate_judge_report`'s structure/range checks (reusing `JudgeIssue` from
`judging.py`, no new dataclass).

### 4. Merge

```python
def apply_semantic_grounding_report(
    report: dict[str, Any],
    semantic_report: dict[str, Any],
) -> dict[str, Any]:
```

Writes, per trial that has a matching item:
`trial["semantic_grounding_check"] = {"score": ..., "claims": [...], "source": "audited_grounding_judge"}`.

Writes at top level:
`report["semantic_grounding"] = {"score": ..., "coverage": ..., "num_judged_trials": ..., "verdict_breakdown": {...}, "source_report": {...}}`.

Does not write to `metric_scores`, does not recompute `overall_score`. A
maintainer who wants to actually re-rank on semantic grounding does that as
a separate, explicit analysis step (out of scope here) — this PR only adds
the measurement.

## CLI

`scripts/run_openvoicecs.py` gains two subcommands mirroring `model-judge` /
`apply-judge-report` flag-for-flag:

```
semantic-grounding <report> --judge provider:model [--judge provider:model ...]
                   [--adjudicator provider:model]
                   [--scenarios data/openvoicecs/scenarios_v0.1.json]
                   --annotations-output PATH --grounding-report-output PATH
                   [--graded-report-output PATH]
                   [--max-output-tokens 200] [--temperature 0.0] [--timeout-seconds 60.0]
                   [--env .env]

apply-semantic-grounding-report <report> <grounding_report> [--output PATH]
```

`cmd_semantic_grounding` follows `cmd_model_judge`'s structure: load report,
call `generate_semantic_grounding_annotations`, write annotations JSONL
(reuse `write_judge_annotations_jsonl`), build + validate the grounding
report, optionally apply it and write the graded report.

## Testing

`tests/unit/test_semantic_grounding.py`, mirroring
`test_judging.py`'s model-judge tests:

- Fake `caller` (no network, no API key — matches CONTRIBUTING's "No API key
  is needed to develop, validate, or test").
- Blinding assertion: payload sent to the judge has no `scenario_id`
  (beyond claim context), no scores, no tool calls, no oracle data.
- `honest_alternative` verdict is not counted as missing (the core fix).
- `not_grounded` is counted as missing.
- Disagreement between two raters triggers the adjudicator; agreement does not.
- `apply_semantic_grounding_report` leaves `metric_scores`, `overall_score`,
  and every existing `trial["grounding_check"]` byte-for-byte unchanged
  (regression guard proving this is additive).
- `validate_report()` (the existing report-contract validator) still passes
  on a report with `semantic_grounding` attached — i.e. the new field
  doesn't break existing report validation, and existing validation doesn't
  need to know about it (`additionalProperties`-style tolerance already
  implicit in how `conversation_experience` was added).

## Docs

`docs/known-limitations.md` section 7 gets a short addendum (not a rewrite —
the literal-matcher critique stays true and stays the default path):

> An opt-in semantic grader is now available: `scripts/run_openvoicecs.py
> semantic-grounding` / `apply-semantic-grounding-report`, following the
> same blinded, audited pattern as the model-judge subjective-quality
> pipeline (`docs/known-limitations.md` section 13). It classifies each
> required claim as grounded, honestly-alternative, or genuinely missing
> via a narrow LLM-judge call, and is merged into a report's
> `semantic_grounding` field without altering `factual_grounding` or
> `overall_score`. Judge-to-human agreement on this classification has not
> been measured yet — treat it as provisional, same caveat as section 13.

## Extensibility (not built now)

`generate_semantic_grounding_annotations`'s `caller: ModelJudgeCaller`
parameter is the seam for a future embedding-similarity backend: a caller
that computes cosine similarity instead of asking a chat model, returning a
`grounded`/`not_grounded` verdict (never `honest_alternative`, since
similarity alone can't detect honest failure framing). No interface change
needed to add it later.

## File list

- `src/evaluation/benchmark/semantic_grounding.py` (new)
- `tests/unit/test_semantic_grounding.py` (new)
- `scripts/run_openvoicecs.py` (add two subcommands + two `cmd_*` functions)
- `docs/known-limitations.md` (section 7 addendum)
- `Makefile` / CONTRIBUTING.md validator list: no change required —
  `semantic-grounding` output is not part of `make check`'s gate (it needs a
  live judge model, same reason `model-judge` isn't in `make check` either).

## PR description (per CONTRIBUTING.md's requirements)

To be filled in at PR time, but the answers are already fixed by this design:

- Validity gap addressed: section 7, `factual_grounding` phrase-matcher
  conflation (synonymy misses + honest-failure penalization).
- Release files changed: none (`data/openvoicecs/**` untouched).
- Scoring behavior changed: no, by default — new field is additive and only
  populated when the new CLI commands are explicitly run and merged.
- Public-dev/sealed-test content moved: no.
- Validation commands: `make check`; `python -m pytest tests/unit/test_semantic_grounding.py`.
- Judge/protocol/study/annotation-package/sealed-ops/external-system/claims
  artifacts changed: no — this is a new, separate pipeline, not a change to
  the existing judge artifacts.
- Contamination/licensing/consent: none — no new scenario or corpus content.
