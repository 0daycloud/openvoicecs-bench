# PLAN.md — factual_grounding Semantic Grader Contribution

Full project plan and progress log, chronological, kept accurate as work
proceeds. See [CLAUDE.md](CLAUDE.md) for the standing session rules this
plan operates under.

## 1. Background

2-day take-home case study. Two task options were offered: (a) a macOS
voice-coding-agent app (Mamachi) requiring Swift/Xcode, (b) improving this
benchmark's grader. Option (a) was ruled out — no Mac, no Swift experience,
Windows machine only, Python experience. Option (b) chosen: Python-only,
cross-platform, well-scoped, and the repo's own docs (known-limitations.md
§7, CONTRIBUTING.md) explicitly flag it as "the single most valuable
contribution available."

## 2. Problem Statement

- `check_factual_grounding` (src/evaluation/benchmark/openvoicecs.py, line
  ~2721) only did literal regex/phrase matching against `required_claims`
  and `forbidden_claims`.
- Concrete failure modes (from known-limitations.md §7): missed valid
  paraphrases (e.g. "no fee" vs "fee waiver"); falsely penalized honest
  failure reports ("couldn't complete, escalated") as ungrounded.
- Removing this metric was documented to reshuffle 36 of 44 models'
  rankings — it distorts the leaderboard.

## 3. Goal / Success Criteria

- Make `factual_grounding` recognize semantically equivalent claims without
  losing the fast, free, deterministic literal-match path for the common
  case.
- Do NOT weaken the offline/no-API-key guarantee for `--agent oracle`.
- Minimum bloat — smallest change that actually fixes the documented
  problem, reusing existing infrastructure wherever possible.

## 4. Architecture & Design Decisions (with rationale)

- **Hybrid cascade (literal first, semantic fallback only on literal
  miss), DEFAULT ON — not an opt-in flag.** An opt-in, default-off feature
  would never actually get used in real sweeps, defeating the point. This
  exact mistake was made mid-project (an earlier draft plan proposed a
  `"phrase"|"semantic"` mode flag defaulting to the old behavior) and was
  corrected before implementation: the hybrid path is the default, and only
  the *old* behavior sits behind an explicit opt-out.
- **`OPENVOICECS_GROUNDING_MODE=legacy` kept only as a comparison/debug
  escape hatch, not the default.** Lets anyone reproduce the pre-semantic
  phrase-matcher scores for comparison without it being what real scoring
  runs use.
- **`forbidden_claims` cascade uses a `_forbidden_claim_near_miss` filter.**
  Without it, "regex didn't match" alone would fire the semantic judge on
  nearly every trial, since a compliant reply not matching a forbidden
  pattern is the normal case. It was empirically tuned directly against the
  oracle agent: an initial single-shared-keyword version produced 56/220
  false triggers, caused by two distinct bugs — vocabulary legitimately
  shared with a required claim (both "refund processed" and forbidden
  "instant refund" contain "refund"), and substring collisions ignoring
  negation ("changed" matching inside "unchanged"). Fixed by requiring
  *every* content keyword of a pattern to appear with word-boundary
  matching, plus keeping digit tokens regardless of length (needed for
  pre-correction-value patterns like "14 Pine"). Re-verified at 0/220 false
  triggers after the fix.
- **Batched single LLM call per trace (not per-claim).** Cost/latency: a
  trace with several unresolved claims should not turn into several
  sequential API calls.
- **Structured JSON judge output, not free text.** Testability and
  auditability — parseable, loggable, deterministically validated verdicts
  (`{"grounded"/"violated", "reason"}`) instead of prose to regex against.
- **`temperature=0` + pinned model (`OPENVOICECS_GROUNDING_JUDGE` env,
  default `openai/gpt-4o-mini`).** Determinism reason — but the honest
  caveat is that the semantic fallback is *not* bit-for-bit deterministic
  even at temp=0, since it's still a live model call. This is disclosed in
  `docs/known-limitations.md` §7.
- **Reused `judging.py`'s existing injectable-caller pattern
  (`ModelJudgeCaller` / `call_openai_compatible_model_judge`) instead of
  writing new provider code.** Minimal-bloat reason — the repo already had
  a tested, working OpenAI-compatible caller abstraction for exactly this
  kind of judge call.
- **`classify_trial_error` handles judge-call failures as `"infrastructure"`
  errors** (trial excluded, not silently scored 0). Consistency with the
  repo's existing v0.2 infra-exclusion mechanism — a judge outage carries no
  information about the agent under test, same reasoning the repo already
  applies to provider billing/network failures.
- **Oracle-offline guarantee.** The oracle always uses literal terms, so
  the cascade never reaches the semantic path for it — locked by dedicated
  regression tests for BOTH the `required_claims` and `forbidden_claims`
  paths separately, since each is gated by a different condition and either
  could regress independently.

## 5. Implementation Log (chronological)

All entries below are dated 2026-08-08 unless noted otherwise.

- [DONE] Environment setup: venv, `pip install -e ".[dev]"`, baseline
  validate/score/pytest all green.
- [DONE] Fixed Windows CRLF/hash mismatch — repo-local
  `core.autocrlf=false` + renormalize, global git config untouched.
- [DONE] Identified and ignored 3 pre-existing Windows path-separator
  pytest failures (release_bundle/release_verification) — unrelated to
  this work, noted, not fixed.
- [DONE] Read and understood `check_factual_grounding`, the data schema
  (`required_claims`/`any_terms`, `forbidden_claims`/`patterns`), and
  existing test patterns in `test_scoring_validity.py`.
- [DONE] Designed and implemented the hybrid cascade grader (see section 4).
- [DONE] Fixed false-positive semantic triggers on the `forbidden_claims`
  path (`_forbidden_claim_near_miss` filter), verified 0/220 false triggers
  on the oracle.
- [DONE] Added `tests/unit/test_factual_grounding.py` (16 tests).
- [DONE] Updated 5 pre-existing tests (`test_openvoicecs.py`,
  `test_scoring_validity.py`) to pin `OPENVOICECS_GROUNDING_MODE=legacy`
  where they used synthetic text unrelated to grounding semantics.
- [DONE] Updated `docs/known-limitations.md` §7 (partly-fixed status,
  determinism note, legacy mode note).
- [DONE] Verified: `pytest tests/unit` → 250 passed, 1 skipped, 3 known
  unrelated fails; `score --agent oracle --trials 1` → 220/220, 100/100,
  no API key used; `ruff` clean on touched files.
- [DONE] `.env` configured locally with `OPENAI_API_KEY` only (gitignored,
  never committed).
- [DONE] Created `CLAUDE.md` with standing session rules (project context,
  locked design decisions, scope discipline, git workflow).
- [DONE] Created this file (`PLAN.md`).
- [DONE] 2026-08-09: `.env`'s `OPENAI_API_KEY` was rotated to a valid key.
  Ran the live smoke test: pulled 10 real trials from
  `data/openvoicecs/runs/requested_v02/reports/openai_gpt_5_6_sol.json`
  whose legacy grader already flagged a `missing_required_claims` or
  `unsupported_claims_detected` gap, and re-scored them with
  `mode="hybrid"` against a real `openai:gpt-4o-mini` judge call.
  **First attempt found a real bug**: 4/10 calls raised
  `RuntimeError: grounding judge call failed: Extra data: line 1 column
  ...`. Root cause: `call_openai_compatible_model_judge`
  (`src/evaluation/benchmark/judging.py`) never set
  `response_format={"type":"json_object"}` on the OpenAI request, so
  gpt-4o-mini would occasionally append stray content after a complete
  JSON object (observed as what looks like a second, partial object),
  which `json.loads` rejects as "Extra data" even though the actual
  object was well-formed. This is a direct dependency of
  `check_factual_grounding`'s semantic fallback (every real judge call
  goes through it), so it was in scope to fix. **Fix**: force
  `response_format={"type":"json_object"}` for `provider=="openai"` only
  (other OpenAI-compatible providers left untouched — not verified to
  support the flag, and the bug was only confirmed on `openai`).
  Verified first that both existing prompts using this caller
  (`_build_grounding_judge_messages` and `_build_model_judge_messages`)
  already contain the literal word "JSON", which OpenAI's `json_object`
  mode requires to be present somewhere in the messages. Re-ran
  `pytest tests/unit` after the fix — same 250 passed / 1 skipped / 3
  known-unrelated-fails as before, no new failures. Re-ran the smoke
  test: **0/10 errors**, 7/10 scores changed from the legacy-only pass
  (paraphrases like "at no charge" → `fee_waived`, "update is complete" →
  `completed_update` correctly grounded; a genuine omission — agent never
  stated a credit amount — correctly still scored not-grounded, so the
  judge isn't just rubber-stamping everything).
- [DONE] 2026-08-09: Before/after comparison. Re-scored all 8 models in
  `data/openvoicecs/runs/requested_v02/reports/` (69 scenarios × 3 trials
  = 207 trials/model, 1656 trials total) with the hybrid grader against a
  real `openai:gpt-4o-mini` judge, keeping every other metric untouched
  and recomputing `overall_score` with the repo's own `METRIC_WEIGHTS`
  formula. Scope note: restricted to this one run set (the smallest
  complete batch, 8 models) rather than all 5 run directories under
  `data/openvoicecs/runs/` (150+ report files combined) — re-scoring the
  full history would mean several thousand live judge calls for
  overlapping/superseded data (`text_action_v02_merged` largely subsumes
  the others); a full-sweep re-score is left as a follow-up if reviewers
  want it before merge. 412 of 1656 trials needed a real judge call (the
  rest resolved by the unchanged literal pass), **0 judge errors**.
  Results (`factual_grounding` mean, `overall_score`):

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

  `factual_grounding` only ever increases (the semantic pass adds
  grounded claims on top of the literal pass, never removes any), as
  expected. **2 of 8 models changed rank**: `openai_gpt_5_6_luna` (6th →
  7th) and `openai_gpt_5_6_terra_pro` (7th → 6th) swapped places — the
  two "terra" variants had by far the worst literal-match grounding
  scores (0.37, 0.40) and the largest gains (+6.87, +10.05 overall
  points), consistent with them producing more heavily paraphrased
  replies that the old literal matcher penalized. This is a real,
  measured number for *this* change on *this* 8-model slice — not the
  old "36 of 44 reshuffled" figure from removing the metric entirely,
  which was a different comparison (metric removed vs. metric kept).
- [DONE] 2026-08-09: Ran the full `make check` gate by hand (no `make` on
  this Windows shell, so its six steps were run individually with
  `.venv/Scripts/python.exe`): `ruff check .` clean; both validity gates
  (`mark_ungrounded_tool_arguments.py --check`,
  `bind_forbidden_event_triggers.py --check`) pass; `validate` (220
  scenarios) pass; `validate-reviews` pass; `validate-submission-intake`
  pass; `validate-release-bundle` pass; `pytest tests/unit` → 250 passed,
  1 skipped, 3 failed. The 3 failures are exactly the pre-existing Windows
  path-separator bug already logged above
  (`test_release_bundle.py::test_build_frontier_release_bundle_writes_valid_artifacts`,
  `test_release_verification.py::test_verify_openvoicecs_seed_release_passes`,
  `test_release_verification.py::test_verify_openvoicecs_release_can_require_audio_assets`)
  — confirmed by diffing a freshly-generated release audit against the
  saved one: every field matches except `path` using `\` instead of `/`.
  Not fixed, per section 7. `verify-release --strict` also fails on the
  same cause (`saved_release_audit` check compares path strings verbatim
  against the committed `release_audit.json`).
- [DONE] 2026-08-09: Wrote the PR description draft — see
  [PR_DRAFT.md](PR_DRAFT.md). Not opened yet; still waiting on the two
  API-key-blocked PENDING items above before this goes out for real.

## 6. Testing & Verification Strategy

- **Unit level:** mocked-caller tests for the semantic path (both grounded
  and ungrounded outcomes), oracle-offline call-count-zero tests for both
  the required and forbidden paths, legacy-mode regression tests.
- **Integration level (pending):** real API smoke test, before/after
  leaderboard comparison using stored run data.
- **Full gate (pending):** `make check` — expands to scenario validation,
  review-manifest validation, submission intake validation, the strict
  release gate, release-bundle verification, and the unit tests (the same
  set CI runs). Command for lint on touched files only (repo carries
  pre-existing lint debt, full-repo lint is not the gate):

  ```bash
  python -m ruff check $(git diff --name-only main...HEAD -- '*.py')
  ```

## 7. Explicitly Out of Scope (do not touch, note only)

- Windows path-separator bug in release_bundle/release_verification tests
  — pre-existing, unrelated, noted in `docs/known-limitations.md` if not
  already, never fixed.
- Any other bug or improvement noticed elsewhere in the codebase.
- AWS Bedrock/Gemma test port, if ever created for cheap large-scale
  testing — must stay local/untracked, never committed.
- Any other provider keys (Anthropic, Google, DeepSeek, etc.) — not needed
  for this task.

## 8. Definition of Done / PR Checklist

Mirrors the repo's own `CONTRIBUTING.md` "Pull Request Requirements"
verbatim — a PR should state:

- what scientific or operational validity gap it addresses;
- which release files changed;
- whether scoring behavior changed;
- whether any public-dev or sealed-test content moved;
- validation commands and results;
- whether judge protocol, judge study, judge annotation package, sealed
  operations, external-system registry, claim package, or release-bundle
  artifacts changed;
- contamination, licensing, and consent implications.

Plus: `make check` passes, oracle stays 220/220 and offline, no unrelated
files touched, no unrelated fixes bundled in.

## 9. Known Limitations To Disclose Honestly In The PR

- Semantic fallback is not bit-for-bit deterministic (temp=0 doesn't fully
  guarantee it for a live model call).
- Adds latency/cost on the subset of claims that fail literal matching —
  quantified on the `requested_v02` slice: 412 of 1656 trials (24.9%)
  triggered a real judge call.
- The `_forbidden_claim_near_miss` pre-filter is a keyword heuristic, not
  full paraphrase detection — a forbidden claim reworded with entirely
  different vocabulary can still slip past both the regex and the
  pre-filter.
- The semantic judge itself has not been evaluated against a labeled
  ground-truth set of grounding verdicts — its accuracy is assumed, not
  measured.
- The hybrid scorer has been run against one 8-model slice
  (`requested_v02`), not the full ~150-report model sweep across all run
  directories — see section 5's 2026-08-09 before/after entry for the
  measured numbers and the reason the scope was limited. A full-sweep
  before/after re-score (replacing the old 0.047–0.323 /
  36-of-44-reshuffled "metric removed entirely" comparison with a real
  "metric kept, hybrid vs. legacy" one across every stored run) is not
  yet available.
- `call_openai_compatible_model_judge` (`judging.py`) had a real bug
  found during this work's own live smoke test — see section 5's
  2026-08-09 entry — fixed by forcing `response_format={"type":
  "json_object"}` for the `openai` provider. That fix has no dedicated
  unit test yet: the repo's existing judge tests all go through the
  injectable `ModelJudgeCaller` and mock it, so nothing currently asserts
  on the raw request payload this function builds. Correctness was
  verified live (0/10 then 0/412 real calls failing after the fix) and
  by the full `pytest tests/unit` suite staying green, but a
  request-payload-mocking regression test would be a reasonable
  follow-up.

---
Keep this file updated as work progresses: flip `[PENDING]` to `[DONE]`
with the date when something is finished, add new entries to section 5 in
order.
