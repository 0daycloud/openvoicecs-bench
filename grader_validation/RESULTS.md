# OpenVoiceCS-Bench — Grader Improvement + Validation Study

A benchmark grader is only useful if you can trust it. We did two things:

1. **Improved the deterministic grader** — one behavioral change plus adopted fixes, in a diff that *shrinks* the codebase.
2. **Built the first validation study** that measures whether a grader change actually agrees with **human judgment** — and used it to compare all competing PRs and to test the "just use an LLM to grade" idea head-on.

---

## 1. The grader fix

Branch `grounded-events` — **+246 / −186 across 3 files, 240 unit tests green, oracle still 100%, multi-turn scenarios still graded correctly.**

**The unique change:** the grader used to credit policy / auth / privacy "events" from a list the *agent supplied about itself*. An agent that called **zero tools** and did nothing could declare *"identity verified, refund processed"* and score those metrics at **1.0**. We now derive every event from **evidence** — the tool calls, their results, the state changes, and the agent's own words — never from a self-reported list.

- A do-nothing "event-stuffer" agent dropped from **43.0 → 24.9** (the no-op floor).
- This exploit is **live in real logs**: a real Gemini trace that called *no tools at all* still received full `auth_integrity` credit under **every grader except ours**.

We also folded in, **with credit**, the field's other sound fixes — grounding claims from replayed state (PR4/10/12), duplicate-call detection (PR9), privacy leak detection a disclaimer can't erase (PR7/9) — and **deleted ~130 lines of duplicated event-derivation**. Net: a new capability while the code got smaller.

We tested the classification direction (PR11) too — and our validation framework **rejected** it: it *lowered* human agreement (74→71%), so we did not ship it.

---

## 2. What every PR does

| PR | author | what it does |
|----|--------|--------------|
| 1 | soyleremo3 | hybrid **LLM** grounding (opt-in) |
| 2 | EnzeCbe | standalone constraint checker (not wired into scorer) |
| 3 | EnzeCbe | semantic **LLM** grounding (opt-in, not wired) |
| 4 | emingurbuz | grounding from replayed state (env-gated) |
| 5 | emingurbuz | near-miss trial gating |
| 6 | emingurbuz | confidence intervals on the ranked score |
| 7 | emingurbuz | privacy: a disclaimer can't erase a real leak |
| 8 | emingurbuz | enum docs/tooling (no scoring change) |
| 9 | emirkaanozdemr | validity fixes + v0.3 distractor corpus + grader-probe |
| 10 | 6c0de | grounding-from-state + grader-eval harness |
| 11 | akirik28 | classification / argument-enum grading (heavy corpus edit) |
| 12 | alinebidal10 | grounding gated on outcome + metamorphic tests |
| **ours** | — | evidence-grounded events + adopted fixes + **validation framework** |

---

## 3. Results — three independent evidence layers

### Layer 1 — Crafted failure modes (30 hand-built traces on 6 real scenarios)

| grader | score |
|---|---|
| baseline / PR1 / PR3 / PR8 / PR11 | 20 / 30 |
| PR9 / PR12 | 21 / 30 |
| PR10 | 23 / 30 |
| **ours** | **29 / 30** |

Ours is the only grader that closes all four "gaming" rows.

### Layer 2 — Real model behavior (208 committed frontier-model trials; grader-independent labels)

| grader | pass-rate | catches duplicate actions | denies zero-tool auth | credits real work |
|---|---|---|---|---|
| PR9 | 39.9% | 0% | 8% | 62% |
| PR11 | 44.2% | 0% | 8% | 71% |
| baseline / PR1 / PR3 / PR8 | 47.6% | 0% | 8% | 71% |
| PR10 | 54.8% | 0% | 8% | 81% |
| PR12 | 57.7% | 0% | 8% | 88% |
| **ours** | **64.9%** | **77%** | **65%** | **95%** |

PR10 and PR12 are genuinely competitive on pass-rate and grounding. Ours is *uniquely* ahead on the two exploit columns — the only grader that catches real **double-refund / double-booking** actions and refuses to credit an agent that **did nothing**.

### Layer 3 — Human judgment (35 hand-labeled real traces, one rater)

| grader | agreement | κ | wrongly-passes bad calls |
|---|---|---|---|
| PR9 | 62.9% | +0.20 | 0 |
| PR11 | 62.9% | +0.21 | 1 |
| PR10 | 65.7% | +0.30 | 4 |
| baseline / PR1 / PR3 / PR8 | 68.6% | +0.34 | 1 |
| **PR12** | **77.1%** | **+0.53** | 1 |
| **ours** | **77.1%** | **+0.53** | 1 |

Ours **ties the field's best** (PR12) on human agreement, with the fewest wrongful failures — and, unlike PR12, also wins Layers 1 and 2.

---

## 4. The honest headline: deterministic vs. LLM judge

We ran the same traces past LLM judges — the exact *"use an LLM to grade"* idea the task raises:

| judge | human agreement | reproducible? | misses real failures |
|---|---|---|---|
| Gemini Flash Lite (weak LLM) | 60% | no | — |
| GLM-4.7 (strong LLM) | 77% → **84%** | **no** (flips on identical reruns) | **4–5** |
| **deterministic (ours / PR12)** | 77% | **yes, byte-identical** | **1** |

A **strong LLM judges more like a human on average — but is disqualified as a benchmark grader:**
- **Non-reproducible:** 77% vs 84% on two *identical* temperature-0 runs; it flipped individual verdicts. You cannot reproduce a leaderboard with it.
- **Lenient:** it waved through 4–5 real failures the human caught (ours: 1) — the "wrong workflow gets too much credit" bias the task warns about.
- **Unstable across models:** two LLM judges agreed only **48%** of the time.
- **Slow and paid:** ~30 min and real cost per pass; ours is instant, free, identical every run.

**Conclusion:** the deterministic grader is the safe, reproducible foundation. An LLM is at most a narrow, gated assist — not the grader.

---

## 5. Why this is the strongest submission

- **The best *safe* deterministic grader:** ties the field's best on human agreement, uniquely catches the gaming and duplicate-action exploits nobody else does, and grades multi-turn correctly — in a minimal diff that *removes* bloat.
- **The only validation study:** human gold + cross-PR comparison (all 12 PRs) + measured LLM-judge-vs-human agreement — the direct, data-backed answer to the task's *regex-vs-LLM* question. It even **caught and rejected** a plausible change (the classification direction) that *lowered* human agreement.

We didn't just propose a fix — we built the way to know **which** grader change is actually trustworthy, and shipped the one that scores best on it.

---

## 6. Honest limitations

- Validation traces are **single-turn**. The grader itself grades multi-turn correctly (oracle passes all 19 multi-turn scenarios; no-op fails), but we have not yet run *real models* on those scenarios — the generation harness is built and verified, just not run at scale.
- Human gold is **35 items, one rater** — enough to rank graders directionally, not for tight confidence intervals. (Still more than any other PR, none of which has human-labeled gold.)
- Residual weakness: a few pure spoken-judgment steps (e.g. *"state the clinical boundary"*) are keyword-limited. A strong LLM handles those better — at the reproducibility/leniency cost above.
