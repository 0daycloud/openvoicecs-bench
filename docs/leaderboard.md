# OpenVoiceCS-Bench Leaderboard — v0.2

**Corpus:** 220 scenarios; `text_to_action` track (69 scenarios), 3 trials each.
**Scorer:** v0.2. **Runs:** `openrouter_top50_text_action_3trial_v02` plus the
`requested_v02` follow-up, merged in `text_action_v02_merged`.
**Artifact:** [`leaderboard.csv`](../data/openvoicecs/runs/text_action_v02_merged/leaderboard.csv)

This replaces the v0.1 sweep, which measured the harness rather than the models
and is **withdrawn** — see [known-limitations](known-limitations.md) section 1.

★ marks the eight models scored in the follow-up run after the original sweep.

## How to read this

- **The podium is trustworthy. Fine-grained mid-table ordering is not.**
  `factual_grounding` is still a literal phrase matcher; removing it leaves the
  leaders intact but reshuffles most of the middle. Gaps under a few points are
  not results.
- **One sweep, three trials, no confidence intervals.** Nothing here separates
  models a couple of points apart.
- **The floor is 24.89, not zero.** The no-op agent — which replies politely and
  never acts — scores 24.89, because "don't do harm" metrics are satisfied by
  inaction. The lowest ranked model (25.38) is barely above it, so the bottom of
  this table is indistinguishable from doing nothing at all.
- **`overall_score` and `task_success` disagree, informatively.** Read both.

## Inclusion rule

A model is ranked only if **at least 90% of its trials produced a usable trace**.
Six of the 58 attempted models are excluded as *unmeasured* — a different
statement from "scored badly":

| Model | Trial coverage | Why |
| --- | ---: | --- |
| `nvidia/nemotron-3-super-120b-a12b:free` | 54% | free-tier rate limiting |
| `nvidia/nemotron-3-ultra-550b-a55b:free` | 52% | free-tier rate limiting |
| `poolside/laguna-m.1:free` | 7% | free-tier rate limiting |
| `nex-agi/nex-n2-pro:free` | 0% | HTTP 404, moved behind a paywall |
| `openai/gpt-oss-120b:free` | 0% | HTTP 404, free tier withdrawn |
| `openrouter/owl-alpha` | 0% | HTTP 404, no endpoints |

This rule is load-bearing. Without it `poolside/laguna-m.1:free` ranked **first**
on 7% coverage, and the two 404 models ranked at 0.00 as though they had been
evaluated and failed.

## Results

| # | Model | Overall | Task | Tool | SOP | Auth | Safety | Grounding* | Tokens/success | p50 (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `anthropic/claude-fable-5` | **88.87** | 0.754 | 0.912 | 0.911 | 0.976 | 0.889 | 0.913 | 9,730 | 20.1 |
| 2 | `moonshotai/kimi-k3` ★ | **86.96** | 0.715 | 0.902 | 0.906 | 0.971 | 0.908 | 0.870 | 5,759 | 18.9 |
| 3 | `openai/gpt-5.6-luna-pro` ★ | **86.15** | 0.860 | 0.867 | 0.841 | 0.845 | 0.947 | 0.826 | 23,211 | 14.0 |
| 4 | `z-ai/glm-5.2` ★ | **81.11** | 0.435 | 0.811 | 0.910 | 0.976 | 0.927 | 0.923 | 7,481 | 9.1 |
| 5 | `anthropic/claude-opus-4.6` | **81.03** | 0.425 | 0.808 | 0.913 | 0.971 | 0.889 | 0.937 | 13,549 | 11.3 |
| 6 | `openai/gpt-5.6-sol` ★ | **78.89** | 0.382 | 0.791 | 0.908 | 0.971 | 0.942 | 0.879 | 8,074 | 10.5 |
| 7 | `google/gemini-3.1-pro-preview` | **77.09** | 0.387 | 0.793 | 0.907 | 0.971 | 0.884 | 0.797 | 15,888 | 9.1 |
| 8 | `deepseek/deepseek-v3.2` | **76.42** | 0.222 | 0.738 | 0.918 | 0.986 | 0.913 | 0.952 | 23,488 | 8.2 |
| 9 | `google/gemini-3.5-flash` | **75.57** | 0.227 | 0.740 | 0.908 | 0.971 | 0.894 | 0.923 | 24,933 | 4.2 |
| 10 | `anthropic/claude-opus-4.7` | **75.36** | 0.295 | 0.762 | 0.913 | 0.981 | 0.908 | 0.816 | 22,632 | 6.5 |
| 11 | `anthropic/claude-sonnet-4.5` | **75.16** | 0.188 | 0.727 | 0.913 | 0.971 | 0.899 | 0.947 | 27,767 | 6.3 |
| 12 | `openai/gpt-5.5` | **75.04** | 0.208 | 0.734 | 0.908 | 0.976 | 0.899 | 0.918 | 21,536 | 5.0 |
| 13 | `anthropic/claude-haiku-4.5` | **74.86** | 0.188 | 0.727 | 0.915 | 0.976 | 0.903 | 0.927 | 28,448 | 3.7 |
| 14 | `openai/gpt-5.6-sol-pro` ★ | **74.80** | 0.222 | 0.738 | 0.908 | 0.971 | 0.947 | 0.879 | 77,784 | 16.6 |
| 15 | `minimax/minimax-m2.5` | **74.60** | 0.184 | 0.720 | 0.916 | 0.986 | 0.908 | 0.918 | 29,927 | 8.3 |
| 16 | `minimax/minimax-m2.7` | **74.16** | 0.237 | 0.731 | 0.907 | 0.971 | 0.903 | 0.855 | 21,625 | 9.1 |
| 17 | `z-ai/glm-5.1` | **74.06** | 0.193 | 0.724 | 0.907 | 0.971 | 0.889 | 0.899 | 25,293 | 8.4 |
| 18 | `z-ai/glm-5` | **73.58** | 0.179 | 0.709 | 0.905 | 0.966 | 0.908 | 0.903 | 28,092 | 11.0 |
| 19 | `openai/gpt-5.3-codex` | **73.48** | 0.188 | 0.716 | 0.890 | 0.952 | 0.889 | 0.903 | 24,792 | 4.8 |
| 20 | `qwen/qwen3.7-max` | **73.46** | 0.191 | 0.728 | 0.910 | 0.971 | 0.887 | 0.863 | 31,217 | 13.7 |
| 21 | `qwen/qwen3.7-plus` | **73.21** | 0.193 | 0.729 | 0.905 | 0.981 | 0.899 | 0.845 | 30,124 | 11.3 |
| 22 | `google/gemma-4-26b-a4b-it` | **73.06** | 0.188 | 0.727 | 0.903 | 0.971 | 0.899 | 0.845 | 27,612 | 4.2 |
| 23 | `z-ai/glm-4.5-air` | **72.93** | 0.179 | 0.712 | 0.904 | 0.966 | 0.899 | 0.879 | 29,721 | 10.9 |
| 24 | `google/gemini-3-flash-preview` | **72.78** | 0.227 | 0.740 | 0.908 | 0.971 | 0.899 | 0.783 | 22,370 | 4.1 |
| 25 | `google/gemini-2.5-flash-lite` | **72.77** | 0.188 | 0.721 | 0.901 | 0.966 | 0.899 | 0.831 | 29,145 | 2.9 |
| 26 | `anthropic/claude-sonnet-4.6` | **72.30** | 0.188 | 0.727 | 0.913 | 0.986 | 0.899 | 0.797 | 28,716 | 6.2 |
| 27 | `deepseek/deepseek-v4-flash` | **72.17** | 0.188 | 0.724 | 0.900 | 0.966 | 0.899 | 0.816 | 25,307 | 11.0 |
| 28 | `moonshotai/kimi-k2.5` | **71.89** | 0.237 | 0.728 | 0.897 | 0.952 | 0.884 | 0.773 | 20,435 | 16.0 |
| 29 | `openai/gpt-5.4` | **71.71** | 0.188 | 0.695 | 0.882 | 0.932 | 0.894 | 0.850 | 22,827 | 3.4 |
| 30 | `google/gemini-3.1-flash-lite-preview` | **71.59** | 0.188 | 0.727 | 0.902 | 0.971 | 0.899 | 0.749 | 26,860 | 1.4 |
| 31 | `google/gemini-3.1-flash-lite` | **71.49** | 0.188 | 0.727 | 0.905 | 0.971 | 0.899 | 0.758 | 26,827 | 2.8 |
| 32 | `google/gemma-4-31b-it` | **70.83** | 0.184 | 0.722 | 0.902 | 0.966 | 0.899 | 0.754 | 27,721 | 10.8 |
| 33 | `openai/gpt-5.4-nano` | **70.59** | 0.155 | 0.659 | 0.871 | 0.899 | 0.908 | 0.870 | 27,155 | 2.8 |
| 34 | `openai/gpt-5-mini` | **70.39** | 0.169 | 0.688 | 0.892 | 0.942 | 0.889 | 0.816 | 28,528 | 14.5 |
| 35 | `qwen/qwen3-235b-a22b-2507` | **70.12** | 0.188 | 0.727 | 0.903 | 0.971 | 0.884 | 0.705 | 24,959 | 4.4 |
| 36 | `minimax/minimax-m3` | **70.05** | 0.174 | 0.706 | 0.903 | 0.961 | 0.889 | 0.744 | 30,419 | 6.6 |
| 37 | `deepseek/deepseek-v4-pro` | **69.94** | 0.184 | 0.704 | 0.895 | 0.952 | 0.894 | 0.749 | 27,089 | 15.0 |
| 38 | `stepfun/step-3.7-flash` | **68.66** | 0.130 | 0.655 | 0.875 | 0.903 | 0.860 | 0.855 | 28,974 | 4.5 |
| 39 | `openai/gpt-5.6-luna` ★ | **68.41** | 0.618 | 0.627 | 0.721 | 0.618 | 0.942 | 0.599 | 4,438 | 4.8 |
| 40 | `openai/gpt-4o-mini` | **68.14** | 0.169 | 0.708 | 0.898 | 0.971 | 0.899 | 0.647 | 27,919 | 3.8 |
| 41 | `inclusionai/ling-2.6-flash` | **67.78** | 0.126 | 0.671 | 0.877 | 0.923 | 0.874 | 0.773 | 35,606 | 4.2 |
| 42 | `google/gemini-2.5-flash` | **67.20** | 0.188 | 0.724 | 0.901 | 0.966 | 0.899 | 0.556 | 26,434 | 3.4 |
| 43 | `anthropic/claude-opus-4.8` | **66.18** | 0.082 | 0.480 | 0.906 | 0.966 | 0.884 | 0.829 | 71,567 | 5.7 |
| 44 | `openai/gpt-5.4-mini` | **65.74** | 0.184 | 0.655 | 0.851 | 0.865 | 0.899 | 0.628 | 23,021 | 2.2 |
| 45 | `moonshotai/kimi-k2.6` | **65.63** | 0.227 | 0.659 | 0.839 | 0.865 | 0.831 | 0.686 | 14,662 | 11.7 |
| 46 | `openai/gpt-5.6-terra-pro` ★ | **63.89** | 0.241 | 0.703 | 0.874 | 0.923 | 0.971 | 0.396 | 69,519 | 9.3 |
| 47 | `mistralai/mistral-nemo` | **61.13** | 0.072 | 0.667 | 0.899 | 0.971 | 0.889 | 0.430 | 67,877 | 3.9 |
| 48 | `openai/gpt-5.6-terra` ★ | **57.18** | 0.232 | 0.564 | 0.774 | 0.720 | 0.976 | 0.372 | 10,753 | 3.6 |
| 49 | `openai/gpt-oss-120b` | **54.01** | 0.097 | 0.510 | 0.806 | 0.797 | 0.865 | 0.387 | 42,502 | 8.9 |
| 50 | `tencent/hy3-preview` | **41.35** | 0.010 | 0.438 | 0.651 | 0.652 | 0.657 | 0.299 | 221,504 | 24.2 |
| 51 | `xiaomi/mimo-v2.5-pro` | **32.44** | 0.048 | 0.309 | 0.439 | 0.420 | 0.444 | 0.377 | 33,156 | 18.4 |
| 52 | `xiaomi/mimo-v2.5` | **25.38** | 0.048 | 0.238 | 0.349 | 0.324 | 0.357 | 0.275 | 25,891 | 19.5 |

\* `factual_grounding` is **provisional** — a literal phrase matcher. See
[known-limitations](known-limitations.md) section 7.

## Reference anchors

| Agent | Overall | pass^k | Safety |
|---|---:|---:|---:|
| `oracle` (correct by construction) | 100.00 | 1.00 | 1.000 |
| `noop` (never acts) | 24.89 | 0.00 | 0.990 |

## What the numbers say

**Resolution and rank disagree, and that is the most useful thing here.**
`gpt-5.6-luna-pro` resolves **86.0%** of scenarios — the highest of any model
measured, well clear of `claude-fable-5`'s 75.4% — yet ranks third, because its
`sop_compliance` (0.841) and `auth_integrity` (0.845) trail a field that mostly
sits above 0.90. Its non-pro sibling shows the same shape more sharply:
`gpt-5.6-luna` is 4th on task success (0.618) and roughly 34th overall, with
`auth_integrity` at 0.618 against a field norm near 0.90.

That is a real deployment signal rather than a scoring artifact. These models get
the customer's problem solved and skip verification and procedure to do it — the
profile of an agent that performs well in a demo and fails an audit. An operator
who cares about resolution rate and an operator who cares about compliance should
read different columns.

**Token efficiency spans 50x**, from 4,438 to 221,504 tokens per resolved call.
`gpt-5.6-luna` is the most efficient model in the set — and ranks mid-table.
`tokens_per_success` is total tokens spent divided by scenarios actually resolved,
so it charges an agent for the budget it burns on failures. Success is defined as
`task_success == 1.0`, deliberately not the strict all-seven-metrics `passed`
gate, which the provisional grounding metric would otherwise drag around.

**"pro" is not uniformly better.** The `gpt-5.6` family spans 57.18 to 86.15 — a
29-point internal range, wider than the gap between most vendors. `luna-pro`
beats `luna` decisively; `terra-pro` beats `terra` only slightly, and both terra
variants sit in the bottom third.

**Safety no longer saturates.** It ranges across the field where v0.1 reported a
median of 0.0 for every model, because argument mismatches were being scored as
safety violations. It must still be read next to `task_success`: the no-op agent
scores 0.990 safety by never acting.

## Reproduce

```bash
# Score one model against the track (needs OPENROUTER_API_KEY)
python scripts/run_openvoicecs.py score-provider \
  --provider openrouter --model google/gemini-2.5-flash-lite \
  --track text_to_action --trials 3 --json-trace --output report.json

# Rebuild this table from the published reports (offline, no API key)
python scripts/build_leaderboard.py \
  data/openvoicecs/runs/text_action_v02_merged/reports \
  --output /tmp/leaderboard.csv
```

A full sweep runs unattended on a GCE VM via `scripts/run_sweep_on_gce.sh up`;
it is resumable, and `OPENVOICECS_MODELS_FILE` points it at a shortlist.

## What this does not cover

- One track of five. `robustness`, `adversarial_compliance`, `audio_to_action`
  and `end_to_end_voice` have no provider sweep yet.
- No hosted sealed evaluator, so these numbers carry no contamination control.
- Latency measures the full JSON action loop (up to 8 sequential provider calls),
  not single-turn voice latency.
- Most scenarios are single-turn; multi-turn behaviour is exercised by a
  19-scenario pilot only ([known-limitations](known-limitations.md) section 11).
