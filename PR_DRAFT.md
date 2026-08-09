# PR Draft: Hybrid semantic fallback for `factual_grounding`

Status: **draft, not opened**. Two verification steps below are blocked on
an API key issue (see "Known limitations / open items"); this will be
opened once those are resolved. Tracked in [PLAN.md](PLAN.md).

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

Quantified before/after impact on stored model runs is **not yet
available** — see "Known limitations / open items."

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
- Adds latency/cost on the subset of claims that fail literal matching.
  Not yet quantified (blocked, see below).
- `_forbidden_claim_near_miss` is a keyword heuristic, not full
  paraphrase detection — a forbidden claim reworded with entirely
  different vocabulary could still slip past both the regex and the
  pre-filter.
- The semantic judge has not been evaluated against a labeled
  ground-truth set of grounding verdicts; its accuracy is assumed from
  design and unit-test coverage, not independently measured.
- **Blocked as of 2026-08-09:** the local `OPENAI_API_KEY` used for this
  work is currently rejected by OpenAI (`401 invalid_api_key`) — key
  format itself is clean (no whitespace/quoting corruption), and `.env`
  loading was independently confirmed to work correctly, so the key
  itself needs to be rotated/replaced. Until then, two verification steps
  are still outstanding and this PR should not be considered final:
  1. A live smoke test of the semantic fallback against real (non-oracle)
     model responses, to confirm the end-to-end API call + JSON parsing
     path works outside of the mocked-caller unit tests.
  2. A before/after comparison re-scoring the stored model runs in
     `data/openvoicecs/runs` with the new grader, to replace the old
     "36 of 44 rankings reshuffled" figure with an actual measured number
     for this change specifically.

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
