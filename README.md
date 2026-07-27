# OpenVoiceCS-Bench

**An open benchmark for voice AI customer-service agents.**

[![License: Apache 2.0](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)
[![Status: v0.1](https://img.shields.io/badge/status-v0.1%20research%20preview-orange.svg)](docs/known-limitations.md)

Most voice-agent evaluation asks whether the agent *sounded* helpful.
OpenVoiceCS-Bench asks whether it actually **resolved the customer's problem
without breaking policy** — and makes you prove it with a replayable trace.

> ### Read this before you cite anything
>
> v0.1 is a **research preview**. The scenario suite and the release-integrity
> tooling are solid. The measurement path between a model and a score is not:
> in the published 50-model sweep, **78% of trials ended in a harness adapter
> error**, and **every forbidden-event safety check in the corpus currently
> passes vacuously**.
>
> The numbers in this repository are published as evidence of those bugs, not
> as a leaderboard. **[docs/known-limitations.md](docs/known-limitations.md)**
> documents each one with the command that reproduces it. Fixing them is the
> project's entire near-term roadmap, and it is where contributions help most.

---

## What it measures

Each scenario is a customer-service call with a defined goal, an initial system
state, a set of replayable tools, and a standard operating procedure. The agent
produces a **trace** — messages, tool calls, and policy events — and the trace
is scored deterministically against an oracle. Nothing is graded on vibes;
subjective quality is a separate, blinded, multi-rater judging pass.

Seven deterministic metrics:

| Metric | Question it answers |
| --- | --- |
| `task_success` | Did the customer's actual problem get resolved? |
| `tool_correctness` | Were the right tools called, with the right arguments, in a valid order? |
| `factual_grounding` | Were claims grounded in retrieved state rather than invented? |
| `sop_compliance` | Was the standard operating procedure followed? |
| `privacy` | Was PII minimized, and was disclosure limited to the verified party? |
| `auth_integrity` | Was identity verified before privileged actions? |
| `safety` | Were forbidden actions refused? |

Plus operational evidence that decides whether an agent is deployable at all:
per-turn latency percentiles, tool-call efficiency, cost from a pinned pricing
snapshot, flake rate across trials, and an `experience_proxy` for conversational
quality.

### The corpus

204 scenarios, all synthetic, across five tracks:

| Track | Count | What it exercises |
| --- | --- | --- |
| `text_to_action` | 64 | Core resolution and tool use from a text transcript |
| `robustness` | 45 | Disfluencies, corrections, interruptions, repair |
| `adversarial_compliance` | 40 | Social engineering, impersonation, prompt injection, PHI extraction |
| `audio_to_action` | 35 | Direct audio input to correct action |
| `end_to_end_voice` | 20 | Full duplex voice interaction |

Six domains carry the load — retail (35), travel (35), fintech sandbox (34),
telecom (33), healthcare admin (33), SaaS support (33) — spanning 40 easy,
81 medium, and 83 hard, with 120 audio variants.

The corpus splits into **public-dev** (development, debugging, reproduction)
and **sealed-test** (never published, for contamination-controlled evaluation).
Split commitments publish counts and hashes without revealing sealed IDs.

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/0daycloud/openvoicecs-bench.git
cd openvoicecs-bench
python -m pip install -e ".[dev]"
```

Everything below runs **fully offline with no API key**:

```bash
# 1. Validate the benchmark package.
python scripts/run_openvoicecs.py validate

# 2. Score the oracle reference agent — should pass 204/204.
python scripts/run_openvoicecs.py score --agent oracle --trials 1

# 3. Run the test suite.
pytest tests/unit -q
```

To score a real model you need a provider key:

```bash
cp .env.example .env      # then fill in one provider; .env is gitignored
python scripts/run_openvoicecs.py score-provider \
  --provider openrouter \
  --model-id openai/gpt-4o-mini \
  --trials 3 \
  --output data/openvoicecs/reports/my_run.json
```

Supported providers: OpenAI, Anthropic, Google, DeepSeek, MiniMax, Moonshot /
Kimi, Alibaba / DashScope, xAI, and any OpenAI-compatible endpoint via
OpenRouter.

## Bring your own agent

Your agent does not have to be a single model call. Implement one function that
takes a scenario and returns a trace:

```bash
# Scaffold a submission adapter, then score it.
python scripts/run_openvoicecs.py init-submission submissions/my_agent.py
python scripts/run_openvoicecs.py submit submissions/my_agent.py:run \
  --name my_agent --provider my_org --model-id my_model \
  --trials 3 --output data/openvoicecs/reports/my_agent.json
```

Working examples: [`examples/openvoicecs_custom_agent.py`](examples/openvoicecs_custom_agent.py),
[`examples/openvoicecs_audio_agent.py`](examples/openvoicecs_audio_agent.py), and
[`examples/openvoicecs_submission_adapter.py`](examples/openvoicecs_submission_adapter.py).

Agents behind an HTTP endpoint or a realtime voice transport are supported too
— see `submit-endpoint` and `load` in
[docs/openvoicecs_bench.md](docs/openvoicecs_bench.md).

### Two ways to collect a trace

Providers with native tool calling use them directly. Chat-only models run a
**stepwise JSON action loop** instead, bounded to 8 rounds, collecting messages
and tool calls turn by turn rather than demanding one perfectly-formed JSON
document. This is what makes the long tail of the OpenRouter catalog scoreable
at all — though see limitation 1, because it is still the largest source of
lost trials.

## Judging subjective quality

Deterministic metrics cannot tell you whether an agent was *pleasant*. A
separate audited judging pass handles that, with the controls you would expect
from a study rather than a vibe check: items are blinded (model identity,
oracle, and scores stripped), at least two independent raters score each item,
and disagreements past a rubric threshold go to a named adjudicator.

```bash
python scripts/run_openvoicecs.py model-judge report.json \
  --judge openrouter:openai/gpt-4o-mini \
  --judge openrouter:google/gemini-2.5-flash \
  --adjudicator openrouter:anthropic/claude-sonnet-4.6 \
  --annotations-output annotations.jsonl \
  --judge-report-output judge_report.json \
  --judged-report-output judged.json
```

Human annotation packages follow the same protocol; see `judge-report` and
`apply-judge-report`.

## Reproducibility and release integrity

This is the part of the project that works properly, and it is deliberately
strict. Benchmarks lose credibility when nobody can tell which artifact
produced which number, so every release artifact is SHA-256 pinned and
cross-validated:

```bash
make verify-release             # full release gate
make validate-release-bundle    # bundle hash verification
```

`verify-release` refuses a release whose scenario suite, audio manifest,
splits, provenance, changelog, baselines, reviews, judge protocol, sealed-ops
manifest, external-system registry, claim package, or datasheet does not match
its recorded hash. `validate-submission-intake` binds a submission's card,
judged report, run manifest, release bundle, registry entry, judging evidence,
and claim package together by hash and byte size — so an official claim cannot
quietly cite a different run than the one that was evaluated.

CI runs the full gate on every push and pull request.

Release artifacts live under `data/openvoicecs/`; the layout is documented in
[docs/openvoicecs_bench.md](docs/openvoicecs_bench.md), and versioning, errata,
and sealed-split policy in [GOVERNANCE.md](GOVERNANCE.md).

## How this is used

OpenVoiceCS-Bench is a **research instrument**. There is no product behind it
and no vendor whose ranking it is designed to flatter. It was built to answer a
question the marketing material around voice agents does not: when a voice
agent is put in front of an SOP, an authentication boundary, and a customer
trying to social-engineer it, what actually happens?

Internally it gets used to:

- **Run broad sweeps** across the OpenRouter catalog to see how the field
  behaves in aggregate rather than cherry-picking a few flagship models. The
  driver is `scripts/run_openrouter_top50_text_action_batch.py`; it is
  resumable, so a sweep can be interrupted and continued.
- **Study failure modes** — the interesting output is not the score, it is
  which policy events fail and why. Most sweeps have taught us more about the
  harness than about the models, which is exactly how
  [docs/known-limitations.md](docs/known-limitations.md) got written.
- **Develop the scoring contract against a known-good agent.** The oracle
  agent passes 204/204 by construction, so any oracle regression means the
  harness broke, not the agent.
- **Keep the corpus honest.** `coverage-plan` reports gaps against a target
  profile and recommends what to author next, so coverage grows deliberately
  instead of drifting toward whatever is easy to write.

## Roadmap

The order is deliberate: **fix validity, then open a leaderboard.** Publishing
rankings from an instrument known to be broken would be worse than publishing
nothing.

**Now — make the measurement trustworthy** ([details](docs/known-limitations.md))

1. Make forbidden events derivable, or replace pattern matching with a semantic
   grader. Until safety checks can fail, nothing else matters.
2. Complete required-event derivation so all 204 scenarios are passable.
3. Separate infrastructure errors from model errors, with explicit
   trial-exclusion and retry policy.
4. Mark ungrounded identifiers as `generated_arguments`.
5. Replace binary all-or-nothing trial gating with a continuous aggregate.
6. Re-run the sweep, and only then publish anything shaped like a ranking.

**Next — `public_beta`.** Wider domain and audio coverage, consented speaker
diversity, externally produced baselines, measured judge-to-human agreement.

**Later — `leaderboard_v1`.** Hosted sealed evaluator, external submission
intake, frozen run manifests, paired statistical comparison with confidence
intervals. The manifests and validators for this already exist and are tested;
they are waiting on a trustworthy instrument, not on code.

## Contributing

The most valuable contributions right now are items 1–4 above — they are
well-scoped, they have reproduction commands, and they unblock everything else.
Scenario contributions are welcome too; the corpus is the most reusable part of
this project.

- [CONTRIBUTING.md](CONTRIBUTING.md) — local checks, scenario authoring rules,
  release-file requirements, PR checklist.
- [GOVERNANCE.md](GOVERNANCE.md) — versioning, errata, split discipline,
  maintainer duties.
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) — Contributor Covenant 2.1.
- [SECURITY.md](SECURITY.md) — private vulnerability reporting.

A note on scope: changes that make a model score better without improving what
the benchmark measures will be declined. That includes loosening oracle checks
to raise pass rates.

## Citation

See [CITATION.cff](CITATION.cff), and please cite the specific release version
along with the release-bundle hash you evaluated against.

## License

- **Code** (`src/`, `scripts/`, `tests/`, `examples/`) — Apache License 2.0. See [LICENSE](LICENSE).
- **Data and artifacts** (`data/openvoicecs/`) — Creative Commons Attribution 4.0 International. See [LICENSE-DATA](LICENSE-DATA).

All scenario data is synthetic. It contains no real customer records, no real
personal data, and no recordings of real people.
