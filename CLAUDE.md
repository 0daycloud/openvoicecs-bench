## Git workflow — ALWAYS DO THIS AUTOMATICALLY
- After every meaningful code change, commit it immediately. Do NOT ask for
  confirmation before committing.
- After every commit, push it immediately. Do NOT ask for confirmation
  before pushing.
- Never wait for my approval to commit or push — treat this as
  pre-approved for the entire duration of this project.
- Only exception: if a change is destructive/irreversible (force push,
  history rewrite, deleting the branch) — for those, ask first.

## Commit message rules
- English, imperative present tense.
- One short summary line (~50-70 chars), plus 1-3 line body explaining
  why (not just what) when it's not obvious.
- Example: "Add semantic fallback for factual_grounding claim matching —
  literal matcher missed paraphrases like 'fee waiver' vs 'no fee'."

## Project context
openvoicecs-bench: a benchmark that automatically grades multi-turn
voice-agent (customer-service bot) transcripts against 220 scenarios, each
with an "oracle" answer key. This is a fork of 0daycloud/openvoicecs-bench,
made as part of a 2-day take-home case study. Work happens on this fork,
then gets proposed back to upstream via PR. The "forked from 0daycloud/..."
attribution on GitHub is correct and intentional — never try to hide or
remove it.

## The actual task: improve check_factual_grounding
- Location: src/evaluation/benchmark/openvoicecs.py, check_factual_grounding
  (~line 2721).
- Problem (documented in the repo's own docs/known-limitations.md §7 and
  CONTRIBUTING.md as "the single most valuable contribution"): the grader
  only did literal regex/phrase matching against required_claims and
  forbidden_claims. It missed paraphrases ("no fee" vs "fee waiver") and
  incorrectly penalized honest failure reports ("couldn't complete,
  escalated") as ungrounded.
- Solution already implemented: a hybrid cascade inside
  check_factual_grounding.
  - Literal/regex match runs FIRST (unchanged, free, fast path).
  - ONLY when literal matching fails does a semantic LLM fallback run.
  - This hybrid behavior is the DEFAULT. The old pure-literal behavior is
    still reachable via OPENVOICECS_GROUNDING_MODE=legacy (env var or
    grounding_mode kwarg) for comparison/debugging only — never make legacy
    the default.
  - forbidden_claims path has a _forbidden_claim_near_miss filter to avoid
    false-positive semantic triggers.
  - All missing claims (required + forbidden) in one trace are batched into
    a SINGLE LLM call, not one call per claim.
  - The judge caller returns structured JSON
    ({"required_claims":[{"id","grounded","reason"}],
      "forbidden_claims":[{"id","violated","reason"}]}), never free text.
  - temperature=0, model pinned via OPENVOICECS_GROUNDING_JUDGE env
    (default openai/gpt-4o-mini).
  - The oracle agent always uses literal terms, so it never triggers the
    semantic fallback — `score --agent oracle` must stay 100% offline,
    API-key-free. This is locked by regression tests for BOTH the
    required_claims and forbidden_claims fallback paths. Never break this.
  - If the judge call fails (network/API error), classify it as an
    "infrastructure" error via classify_trial_error and exclude the trial —
    never silently score it as 0.
  - Reused the existing injectable-caller pattern from judging.py
    (ModelJudgeCaller / call_openai_compatible_model_judge) instead of
    writing new provider-call code.

## Environment notes (already resolved, do not redo)
- Windows dev machine. CRLF/hash mismatch was fixed with a REPO-LOCAL
  `git config core.autocrlf false` + renormalize. Global git config was
  never touched.
- 3 pre-existing pytest failures in test_release_bundle.py /
  test_release_verification.py are a pre-existing Windows path-separator
  bug, UNRELATED to this work. Do not fix them, do not spend time on them.
- `.env` (gitignored) holds only OPENAI_API_KEY. Never read, print, log,
  commit, or paste its contents anywhere.

## Scope discipline — most important rule
- Stay strictly inside check_factual_grounding and its direct dependencies.
- If you notice an unrelated bug or possible improvement elsewhere, do NOT
  fix it. Note it in docs/known-limitations.md or mention it to me, and
  stop there.
- No speculative refactors, no unrelated cleanups, no scope creep — this
  case study explicitly penalizes unnecessary/bloated AI-generated code.
  Prefer the simplest solution that solves the problem.
- pytest tests/unit must stay green (except the known Windows
  path-separator exception above); ruff must stay clean; 220 scenarios
  must keep validating.

## GitHub conventions
- One feature = one branch = one PR. Current branch:
  semantic-grounding-grader.
- Never push directly to 0daycloud/openvoicecs-bench (upstream) — only to
  this fork (soyleremo3/openvoicecs-bench), then open a PR when ready.
- PR descriptions should match the repo's own documentation voice: honest,
  measured, evidence-based, states limitations rather than overclaiming.

## Recent changes log
Keep a short running list here, newest on top, max 10 entries, one line
each. Drop the oldest when it exceeds 10 — full history lives in git log,
this is just a quick-glance summary.
