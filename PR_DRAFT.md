# PR Draft: Hybrid semantic fallback for `factual_grounding`

Status: **draft, not opened**. Both live-verification steps are now
complete (see "Live smoke test" and "Before/after comparison" below).
Tracked in [PLAN.md](PLAN.md).

---

## Title

Add hybrid semantic fallback to `factual_grounding` scoring

## What validity gap this addresses

`check_factual_grounding` (`src/evaluation/benchmark/openvoicecs.py`,
~line 2721) previously did pure literal regex/phrase matching against
`required_claims` and `forbidden_claims`. This is documented in the
repo's own `docs/known-limitations.md` §7 and flagged in
`CONTRIBUTING.md` as the single most valuable open contribution.

Concrete failure modes it caused:

- Valid paraphrases were scored as ungrounded (e.g. "no fee" vs the
  required phrasing "fee waiver").
- Honest failure reports ("couldn't complete, escalated") were
  incorrectly penalized as ungrounded instead of being recognized as
  accurate.
- Because `factual_grounding` is a heavily-weighted metric,
  `docs/known-limitations.md` records that removing it entirely reshuffled
  36 of 44 models' leaderboard rankings — i.e. the literal-only matcher's
  errors were large enough to move the leaderboard, not just cosmetic.

## What changed

- `check_factual_grounding` now runs a **hybrid cascade**:
  1. The existing literal/regex matcher runs first, unchanged — free,
     fast, deterministic, and still the path that resolves the vast
     majority of claims.
  2. Only for whatever literal matching leaves unresolved (required
     claims it could not confirm; forbidden claims that share every
     content word with a pattern via word-boundary matching without
     matching it exactly) does a semantic LLM judge run, as a **single
     batched call per trace** — not one call per claim.
- This hybrid behavior is the **default**, not opt-in. The old pure-literal
  behavior is still reachable via `OPENVOICECS_GROUNDING_MODE=legacy` (env
  var or `grounding_mode` kwarg), kept only as a comparison/debug escape
  hatch.
- A `_forbidden_claim_near_miss` filter guards the forbidden-claims path
  against false-positive semantic triggers (tuned empirically against the
  oracle agent to 0/220 false triggers — see commit `41698e9` for the two
  bugs found and fixed along the way: shared vocabulary between a required
  and a forbidden claim, and substring collisions ignoring negation).
- The judge returns structured JSON
  (`{"required_claims":[{"id","grounded","reason"}],
  "forbidden_claims":[{"id","violated","reason"}]}`), never free text.
  `temperature=0`, model pinned via `OPENVOICECS_GROUNDING_JUDGE`
  (default `openai/gpt-4o-mini`).
- Reused the existing injectable-caller pattern from `judging.py`
  (`ModelJudgeCaller` / `call_openai_compatible_model_judge`) instead of
  adding new provider-call code.
- If the judge call itself fails (network/API error), the trial is
  classified as an `"infrastructure"` error via `classify_trial_error` and
  excluded — never silently scored as 0.
- **`score --agent oracle` stays fully offline and deterministic.** The
  oracle always uses literal terms, so the cascade never reaches the
  semantic path for it. Locked by dedicated regression tests for both the
  `required_claims` and `forbidden_claims` fallback paths (they're gated
  by different conditions and could regress independently).
- **Bug fix in a direct dependency, found by this PR's own live smoke
  test:** `call_openai_compatible_model_judge`
  (`src/evaluation/benchmark/judging.py`) is the shared caller this
  feature (and the existing subjective-quality model-judge feature) uses
  to talk to OpenAI-compatible endpoints. It never set
  `response_format={"type":"json_object"}` on OpenAI requests, so
  `gpt-4o-mini` would occasionally append stray content after an
  otherwise well-formed JSON object, which the response parser rejected
  as `json.JSONDecodeError: Extra data`. Fixed by forcing that flag for
  `provider=="openai"` only (left untouched for the other
  OpenAI-compatible providers this caller also serves, since the bug was
  only confirmed against `openai` and the flag's support elsewhere wasn't
  verified). Both existing prompts that go through this caller already
  contain the literal word "JSON" somewhere in their messages, which
  OpenAI's `json_object` mode requires — verified before making the
  change. See "Live smoke test" below for the before/after failure rate.

## Which release files changed

None. This is a scoring-logic change to `check_factual_grounding` plus
tests and docs; no scenario corpus, manifest, judge protocol/study,
sealed-ops, external-system, claims, or release-bundle artifact was
touched.

## Does scoring behavior change?

**Yes, for non-oracle agents**, by construction — that is the point of
the change. `factual_grounding` scores for real (non-oracle) model
responses can now differ from the pre-change literal-only scores whenever
a response paraphrases a required claim or a forbidden claim narrowly
misses the literal pattern. `score --agent oracle` scores are unchanged
(220/220, same as before — verified, see below).

Quantified on one 8-model slice — see "Before/after comparison" below:
`factual_grounding` only ever increases (the semantic pass adds grounded
claims on top of the literal pass, it never removes any), and 2 of 8
models' overall rank changed.

## Live smoke test

Run 2026-08-09 against real (non-mocked) trials pulled from a stored
report (`data/openvoicecs/runs/requested_v02/reports/openai_gpt_5_6_sol.json`),
using a real `openai:gpt-4o-mini` judge call:

- **First pass, 10 trials with a literal-match gap:** 6/10 correctly
  reclassified as grounded (paraphrases like "at no charge" →
  `fee_waived`, "update is complete" → `completed_update`); **4/10 raised
  a JSON-parsing error** — this is what surfaced the
  `response_format` bug described above.
- **After the fix, same 10 trials:** **0/10 errors.** 7/10 scores changed
  from the legacy-only pass. One case (`telecom-billing-credit-001`) was
  correctly left ungrounded — the agent never stated a credit amount, a
  genuine omission, not a wording mismatch — confirming the judge isn't
  simply rubber-stamping every claim it's asked about.

## Before/after comparison

Re-scored all 8 models in `data/openvoicecs/runs/requested_v02/reports/`
(69 scenarios × 3 trials = 207 trials/model, 1656 trials total) with the
hybrid grader against a real `openai:gpt-4o-mini` judge, keeping every
other metric untouched and recomputing `overall_score` with the repo's
own `METRIC_WEIGHTS` formula. 412 of 1656 trials (24.9%) needed a real
judge call; **0 judge errors**.

Scope note: this run set is the smallest complete batch under
`data/openvoicecs/runs/` (8 models). The other 4 run directories total
150+ report files and substantially overlap with each other
(`text_action_v02_merged` largely subsumes the rest) — re-scoring all of
them would mean several thousand live judge calls against mostly
redundant data. A full-sweep re-score is a reasonable follow-up before
merge if reviewers want the complete picture; happy to run it.

| model | grounding before → after | overall before → after |
|---|---|---|
| moonshotai_kimi_k3 | 0.8696 → 0.9130 | 86.96 → 87.83 |
| openai_gpt_5_6_luna | 0.5990 → 0.6087 | 68.41 → 68.61 |
| openai_gpt_5_6_luna_pro | 0.8261 → 0.8406 | 86.15 → 86.44 |
| openai_gpt_5_6_sol | 0.8792 → 0.9469 | 78.89 → 80.24 |
| openai_gpt_5_6_sol_pro | 0.8792 → 0.9565 | 74.80 → 76.35 |
| openai_gpt_5_6_terra | 0.3720 → 0.7150 | 57.18 → 64.05 |
| openai_gpt_5_6_terra_pro | 0.3961 → 0.8986 | 63.89 → 73.94 |
| z_ai_glm_5_2 | 0.9227 → 0.9372 | 81.11 → 81.40 |

**2 of 8 models changed rank:** `openai_gpt_5_6_luna` (6th → 7th) and
`openai_gpt_5_6_terra_pro` (7th → 6th) swapped. The two "terra" variants
had by far the worst literal-match grounding scores (0.37, 0.40) and the
largest gains (+6.87, +10.05 overall points) — consistent with those
models producing more heavily paraphrased replies that the old literal
matcher penalized. This is a real, measured number for *this specific
change* (hybrid vs. legacy, metric kept) on this 8-model slice — not
directly comparable to the old "36 of 44 rankings reshuffled" figure,
which measured something different (removing the metric entirely vs.
keeping it).

## Did any public-dev or sealed-test content move?

No.

## Validation commands and results

Run 2026-08-09 (Windows, no `make` on this shell — the six `make check`
steps run individually via `.venv/Scripts/python.exe`):

```bash
python -m ruff check .
python scripts/mark_ungrounded_tool_arguments.py --check
python scripts/bind_forbidden_event_triggers.py --check
python scripts/run_openvoicecs.py validate
python scripts/run_openvoicecs.py validate-reviews --review-manifest data/openvoicecs/scenario_reviews_v0.1.json
python scripts/run_openvoicecs.py validate-submission-intake --intake data/openvoicecs/submissions/reference_submission_intake_v0.1.json
python scripts/run_openvoicecs.py verify-release --strict ...   # (full flags in Makefile's verify-release target)
python scripts/run_openvoicecs.py validate-release-bundle data/openvoicecs/releases/frontier_seed/release_bundle.json
python -m pytest tests/unit -q
```

Results:

- `ruff check .` — clean.
- Both scenario validity gates — pass.
- `validate` — 220/220 scenarios, audio manifest valid.
- `validate-reviews` — pass.
- `validate-submission-intake` — pass (7/7 artifacts).
- `validate-release-bundle` — pass.
- `pytest tests/unit` — **250 passed, 1 skipped, 3 failed.**
- `verify-release --strict` — fails on one check
  (`saved_release_audit`).

The 3 pytest failures and the 1 `verify-release` failure are all the
**same pre-existing, unrelated issue**: on Windows, `pathlib` serializes
manifest/artifact paths with `\` instead of `/`, so string-equality
checks against POSIX-style paths baked into fixtures/saved artifacts
fail even though the underlying content (hashes, byte counts) is
identical — confirmed by diffing a freshly regenerated release audit
against the committed one field-by-field: every `sha256`/`bytes` value
matches, only the `path` separator differs. This was already identified
and logged as out-of-scope in `CLAUDE.md`/`PLAN.md` before this change
started, and this PR does not touch it.

`score --agent oracle --trials 1` (offline check, no API key used):
220/220 scenarios, unchanged.

## Judge protocol / study / annotation package / sealed ops / external
## systems / claims / release-bundle artifacts

Not changed. The grounding judge reuses the existing generic
`judging.py` caller abstraction; no new judge-protocol document was
introduced for it and none of the sealed/external/claims artifacts were
touched.

## Contamination, licensing, and consent implications

None. No new data was added to the scenario corpus, audio manifest, or
any sealed/held-out split. The change is scoring-logic-only, operating
on existing scenario `required_claims`/`forbidden_claims` fields and
model-generated trial text that is already part of the existing
evaluation flow.

## Known limitations / open items (disclosed, not hidden)

- The semantic fallback is **not bit-for-bit deterministic** even at
  `temperature=0`, since it is still a live model call. Noted in
  `docs/known-limitations.md` §7.
- Adds latency/cost on the subset of claims that fail literal matching —
  quantified: 412 of 1656 trials (24.9%) in the before/after slice.
- `_forbidden_claim_near_miss` is a keyword heuristic, not full
  paraphrase detection — a forbidden claim reworded with entirely
  different vocabulary could still slip past both the regex and the
  pre-filter.
- The semantic judge has not been evaluated against a labeled
  ground-truth set of grounding verdicts; its accuracy is assumed from
  design and unit-test coverage, not independently measured.
- The before/after comparison covers one 8-model slice, not the full
  historical sweep across all 5 run directories — see "Before/after
  comparison" above for the scope reasoning.
- The `response_format` fix in `call_openai_compatible_model_judge` has
  no dedicated unit test — the repo's existing judge tests all go
  through the injectable `ModelJudgeCaller` and mock it, so nothing
  currently asserts on the raw request payload this function builds.
  Correctness was verified live (0/10, then 0/412 real calls failing
  after the fix) and by the full `pytest tests/unit` suite staying green,
  but a request-payload-mocking regression test would be a reasonable
  follow-up.

## Testing

- `tests/unit/test_factual_grounding.py` (new, 16 tests): mocked-caller
  coverage for both grounded/ungrounded semantic outcomes on the
  required-claims path and the forbidden-claims path, oracle-offline
  call-count-zero tests for both paths separately, and legacy-mode
  regression tests reproducing the old pure-literal scores.
- `tests/unit/test_openvoicecs.py`, `tests/unit/test_scoring_validity.py`:
  updated to pin `OPENVOICECS_GROUNDING_MODE=legacy` where they exercise
  synthetic text unrelated to grounding semantics, so they keep testing
  what they were written to test.
