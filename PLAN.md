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
- [PENDING] Live smoke test: run the semantic fallback against a handful
  of real (non-oracle-style) responses with a real OpenAI call, confirm the
  API call + JSON parsing actually works end-to-end (only tested via mocked
  caller so far). Blocked 2026-08-09: the local `OPENAI_API_KEY` is
  currently invalid (`401 invalid_api_key` from OpenAI itself — key format
  is clean, no whitespace/quoting issue, `.env` loading path confirmed
  working). Waiting on a replacement key.
- [PENDING] Before/after comparison: re-score existing stored model runs
  (`data/openvoicecs/runs`) with the new grader, quantify how many of the
  36/44 previously-reshuffled rankings actually change and by how much.
  Blocked on the same invalid API key as above.
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
- Adds latency/cost on the subset of claims that fail literal matching
  (quantify if possible after the before/after comparison in section 5).
- The `_forbidden_claim_near_miss` pre-filter is a keyword heuristic, not
  full paraphrase detection — a forbidden claim reworded with entirely
  different vocabulary can still slip past both the regex and the
  pre-filter.
- The semantic judge itself has not been evaluated against a labeled
  ground-truth set of grounding verdicts — its accuracy is assumed, not
  measured.
- The hybrid scorer has not yet been run across the full model sweep, so
  an updated leaderboard-impact number (replacing the old 0.047–0.323 /
  36-of-44-reshuffled figures) is not yet available.

---
Keep this file updated as work progresses: flip `[PENDING]` to `[DONE]`
with the date when something is finished, add new entries to section 5 in
order.
