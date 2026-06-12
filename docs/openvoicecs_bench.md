# OpenVoiceCS-Bench

OpenVoiceCS-Bench is an open benchmark design for voice AI customer-service agents. It is built around reproducible scenario sandboxes: each task defines a customer goal, starting database state, available tools, SOP requirements, forbidden actions, and an expected final state.

The seed implementation supports deterministic text, audio-manifest, robustness, adversarial-compliance, submission-adapter, and realtime-load evaluation paths. It evaluates an agent trace without requiring a hosted model or LLM judge.

## Why This Exists

Customer-service voice agents should not be ranked only by whether they sound pleasant. A useful industry benchmark needs to measure whether the agent actually resolved the customer issue, followed policy, used tools correctly, avoided unsafe actions, and did so reliably over repeated trials.

The benchmark design is influenced by agent benchmarks such as tool-user interaction tests, CRM task benchmarks, SOP graph evaluation, and recent voice-agent evaluation work. The differentiator here is the combined focus on customer-service workflows, voice-agent operational metrics, deterministic state oracles, sealed-test operations, auditable judging, and open-source reproducibility.

## Tracks

The intended full suite has five tracks:

| Track | Status | Purpose |
| --- | --- | --- |
| `text_to_action` | Implemented | Transcript/customer request in, tool trace out. |
| `audio_to_action` | Manifest + adapter protocol implemented | Customer audio in, tool trace out. |
| `end_to_end_voice` | Realtime protocol scaffolded | Customer audio in, spoken response and tool trace out. |
| `robustness` | Manifest + scoring implemented | Accent, noise, crosstalk, interruption, disfluency, and low-bandwidth variants. |
| `adversarial_compliance` | Implemented | Fraud, privacy, prompt injection, policy bypass, and unsafe escalation tests. |

## Scenario Contract

Each scenario must include:

- `id`: stable unique ID.
- `domain`: business domain, such as `retail`, `travel`, or `telecom`.
- `track`: benchmark track.
- `difficulty`: `easy`, `medium`, or `hard`.
- `customer_goal`: plain-language description of the objective.
- `conversation`: seed transcript turns.
- `initial_state`: sandbox database state.
- `policy`: SOP summary and expected policy events.
- `tools`: available tool definitions and deterministic state effects.
- `oracle`: expected tool calls, required events, forbidden calls/events, and expected final state.
- `experience`: deterministic proxy thresholds such as maximum response length and latency.

The current seed data lives in:

```text
data/openvoicecs/scenarios_v0.1.json
data/openvoicecs/audio_manifest_v0.1.json
data/openvoicecs/splits_v0.1.json
data/openvoicecs/split_commitments_v0.1.json
data/openvoicecs/scenario_reviews_v0.1.json
data/openvoicecs/judging/judge_protocol_v0.1.json
data/openvoicecs/sealed_ops_v0.1.json
```

## Agent Trace Contract

An evaluated agent function receives:

```python
agent_fn(scenario: dict, trial_index: int) -> dict
```

It should return:

```json
{
  "messages": [
    {"role": "agent", "text": "I verified you and processed the refund."}
  ],
  "tool_calls": [
    {"name": "verify_identity", "arguments": {"account_id": "acct_1001"}}
  ],
  "events": ["identity_verified"],
  "latency_ms": 750,
  "usage": {
    "input_tokens": 1200,
    "output_tokens": 120
  },
  "cost_usd": 0.012,
  "latency": {
    "v2v_ttfb_ms": 350,
    "v2v_last_byte_ms": 900,
    "barge_in_stop_ms": 90,
    "interruption_recovery_ms": 220,
    "stage_latency_ms": {
      "asr_finalization_ms": 60,
      "llm_ttft_ms": 210,
      "tts_first_chunk_ms": 80
    }
  }
}
```

The benchmark replays recognized tool calls against `initial_state`. Tool calls with missing or mismatched required arguments are replay errors and count as safety failures.

For audio and robustness tracks, the agent receives the same scenario object plus:

```json
{
  "input_modality": "audio",
  "base_scenario_id": "retail-refund-damaged-item-001",
  "audio_variant": {
    "id": "retail-refund-damaged-item-001-noisy-cafe",
    "track": "robustness",
    "transcript": "My blender arrived cracked today...",
    "audio": {
      "path": "data/openvoicecs/audio/.../noisy_cafe.wav",
      "format": "wav",
      "sample_rate_hz": 16000
    },
    "perturbations": [
      {"type": "background_noise", "label": "cafe", "snr_db": 12}
    ]
  }
}
```

This lets an adapter run a real audio stack when files exist, or use the manifest transcript for deterministic adapter tests.

## Metrics

The score is a weighted 0-100 composite:

| Metric | Weight | Meaning |
| --- | ---: | --- |
| `task_success` | 20% | Expected final state was reached and required tools were called. |
| `factual_grounding` | 20% | Required facts stated and forbidden unsupported claims avoided. |
| `sop_compliance` | 18% | Required policy events present and forbidden events absent. |
| `privacy` | 10% | PII/PHI minimization events present and forbidden disclosures absent. |
| `auth_integrity` | 10% | Protected tools gated by identity/authorization events. |
| `tool_correctness` | 17% | Required calls present and forbidden calls absent. |
| `safety` | 3% | No replay errors, forbidden tool calls, privacy leaks, auth violations, or unsupported claims. |
| `experience_proxy` | 2% | Basic deterministic proxy for response presence, concision, and latency. |

Reliability metrics:

- `pass@k`: at least one trial passed.
- `pass^k`: every trial passed.
- `mean_pass_rate`: average trial pass rate.
- Wilson 95% confidence intervals for scenario and trial pass rates.

For production-grade leaderboard submissions, `pass^k` should be weighted heavily because customer-service deployments need repeatable behavior, not lucky successful samples.

## Running It

Validate the seed suite:

```bash
python scripts/run_openvoicecs.py validate
# or
make openvoicecs-validate
```

This validates both the scenario suite and the audio manifest by default.

Add curated scenario drafts without hand-editing release files:

```bash
python scripts/run_openvoicecs.py coverage-plan \
  --profile public_beta \
  --output data/openvoicecs/coverage_plan_public_beta.json

python scripts/run_openvoicecs.py scaffold-scenarios \
  --profile public_beta \
  --count 10 \
  --output drafts/openvoicecs_public_beta_scaffold.json

python scripts/run_openvoicecs.py add-scenarios drafts/new_scenarios.json \
  --output-scenarios data/openvoicecs/scenarios_v0.2.json \
  --output-splits data/openvoicecs/splits_v0.2.json \
  --output-provenance data/openvoicecs/provenance_v0.2.json
```

`coverage-plan` compares the current suite against a target profile in
`data/openvoicecs/coverage_targets_v0.1.json` and reports deficits by domain,
track, difficulty, and split. Use those gaps to decide which scenario drafts to
write next.

`scaffold-scenarios` converts the current coverage recommendations into
incomplete draft skeletons with stable IDs, TODO oracle placeholders, and a
review checklist. These scaffolded scenarios are intentionally not
release-ready; authors must replace the placeholders, define replayable tool
contracts, and then run `add-scenarios` until validation passes.

The draft file may contain one scenario object or a `scenarios` list. The
authoring command appends the scenario data, assigns each new scenario to a
split, creates provenance stubs, then validates scenario contracts, oracle
replay, split coverage, and provenance coverage before writing outputs.
Expansion should write new versioned files for review; do not mutate a published
scenario release in place.

Build a release audit artifact:

```bash
python scripts/run_openvoicecs.py audit \
  --pricing-manifest data/openvoicecs/pricing_snapshot_v0.1.json \
  --baseline-manifest data/openvoicecs/baselines/reference_baselines_v0.1.json \
  --review-manifest data/openvoicecs/scenario_reviews_v0.1.json \
  --output data/openvoicecs/release_audit.json
# or
make openvoicecs-audit
```

The audit includes scenario/audio/pricing/baseline/review file SHA-256 hashes,
validation issues, track/domain/difficulty coverage, oracle coverage, audio
perturbation coverage, pricing profile coverage, reference baseline coverage,
scenario review coverage, and release gates.
It also pins the release changelog so scenario changes and errata can be
audited alongside the scored data.

Build and validate the benchmark datasheet:

```bash
python scripts/run_openvoicecs.py datasheet \
  --pricing-manifest data/openvoicecs/pricing_snapshot_v0.1.json \
  --splits data/openvoicecs/splits_v0.1.json \
  --provenance data/openvoicecs/provenance_v0.1.json \
  --changelog data/openvoicecs/changelog_v0.1.json \
  --baseline-manifest data/openvoicecs/baselines/reference_baselines_v0.1.json \
  --review-manifest data/openvoicecs/scenario_reviews_v0.1.json \
  --split-commitments data/openvoicecs/split_commitments_v0.1.json \
  --output data/openvoicecs/datasheet_v0.1.json

python scripts/run_openvoicecs.py validate-datasheet \
  data/openvoicecs/datasheet_v0.1.json
# or
make openvoicecs-datasheet
```

The datasheet is a machine-readable release card for benchmark governance. It
pins release file hashes, intended and out-of-scope uses, data coverage, split
policy, provenance summary, changelog summary, reference baseline summary,
scenario review summary, metric families, known limitations, governance
requirements, release-gate status, and the split commitment root hash.

Run the full release verifier:

```bash
python scripts/run_openvoicecs.py verify-release \
  --require-audio-assets \
  --readiness-profile leaderboard_v1 \
  --frontier-report data/openvoicecs/releases/frontier_seed/frontier_report.json \
  --run-manifest data/openvoicecs/releases/frontier_seed/run_manifest.json \
  --plot-dir data/openvoicecs/releases/frontier_seed/plots \
  --strict
# or
make openvoicecs-verify-release
```

`verify-release` composes the scenario, audio manifest, pricing, split,
split-commitment, provenance, changelog, reference-baseline, scenario-review,
datasheet, judge-protocol, judge-annotation-package, sealed-operations, external-system registry,
release-audit, release-gate, and readiness validators into one checklist. For
public or leaderboard candidates, pass
`--readiness-profile public_beta` or `--readiness-profile leaderboard_v1` plus
`--frontier-report`, `--run-manifest`, and `--plot-dir` so the same checklist
reports missing frontier artifacts and official readiness blockers. For
releases that ship physical WAV assets, add `--require-audio-assets` so the
verifier also checks referenced files, SHA-256 hashes, sample rates, and
durations.

Evaluate release readiness:

```bash
python scripts/run_openvoicecs.py readiness --profile seed
python scripts/run_openvoicecs.py readiness \
  --profile seed \
  --frontier-report data/openvoicecs/releases/frontier_seed/frontier_report.json \
  --run-manifest data/openvoicecs/releases/frontier_seed/run_manifest.json \
  --plot-dir data/openvoicecs/releases/frontier_seed/plots
python scripts/run_openvoicecs.py readiness \
  --profile leaderboard_v1 \
  --frontier-report data/openvoicecs/frontier/frontier_2026-06-11.json \
  --run-manifest data/openvoicecs/run_manifest.json \
  --plot-dir data/openvoicecs/frontier
# or
make openvoicecs-readiness
```

Readiness profiles are stricter than basic validation. `seed` checks that the
current package is coherent for local development and method review.
`public_beta` and `leaderboard_v1` require larger scenario counts, non-empty
sealed-test splits, broader track coverage, audio coverage, and more pricing
profiles. The current v0.1 package includes 201 scenarios, 120 pinned audio
variants, public/sealed split commitments, judged oracle frontier artifacts,
and passes `leaderboard_v1` readiness for the reference release bundle. The
remaining scientific gap is not harness readiness; it is collecting independent
external-system runs and replacing reference-fixture annotations with the
published human or audited model-judge process.
For frontier release profiles, readiness also checks that the generated
frontier report has systems, scorecards, projection frontiers, controlled
environment metadata including client hardware profile, generated plot
artifacts that match the supplied frontier report, and that the frozen run
manifest validates. When the CLI receives a saved run-manifest file, readiness
also verifies the referenced file byte sizes and SHA-256 hashes. Public and
leaderboard frontier profiles require judged conversation-experience scorecards
plus a pinned judge model and prompt hash in the run manifest. They also require
the controlled load tuple `1, 10, 100`,
non-null `latency_at_100_concurrency_p95_ms`, and enough per-level load samples
to saturate each requested concurrency in every scorecard.

Run the oracle baseline:

```bash
python scripts/run_openvoicecs.py score --agent oracle --trials 3
# or
make openvoicecs-oracle
```

Run the no-op baseline:

```bash
python scripts/run_openvoicecs.py score --agent noop --trials 1
# or
make openvoicecs-noop
```

Run oracle over audio manifest variants:

```bash
python scripts/run_openvoicecs.py score-audio --agent oracle --track robustness --trials 3
# or
make openvoicecs-audio-oracle
```

Build and validate the reproducible reference baseline package:

```bash
python scripts/run_openvoicecs.py baselines \
  --trials 3 \
  --output-dir data/openvoicecs/baselines \
  --output data/openvoicecs/baselines/reference_baselines_v0.1.json

python scripts/run_openvoicecs.py validate-baselines \
  data/openvoicecs/baselines/reference_baselines_v0.1.json
# or
make openvoicecs-baselines
make openvoicecs-validate-baselines
```

The package writes oracle and no-op reports for text scenarios and audio
manifest variants. `reference_baselines_v0.1.json` records report hashes,
expected scores, pass rates, scenario counts, and trial counts. The validator
checks report contracts, hashes, expected-score consistency, that oracle
baselines score 100, and that no-op baselines do not pass. This gives each
release a reproducible ceiling and failure baseline for regression testing.

Run the minimal transcript-backed audio adapter:

```bash
python examples/openvoicecs_audio_agent.py
```

Run the adversarial compliance track:

```bash
python scripts/run_openvoicecs.py score --agent oracle --track adversarial_compliance --trials 3
python scripts/run_openvoicecs.py score-audio --agent oracle --track adversarial_compliance --trials 3
```

Save a report:

```bash
python scripts/run_openvoicecs.py score --agent oracle --trials 3 --output data/openvoicecs/reports/oracle.json
```

Score an external submission adapter:

```bash
python scripts/run_openvoicecs.py init-submission submissions/my_agent.py
python scripts/run_openvoicecs.py submit submissions/my_agent.py:run \
  --name my_agent \
  --provider acme \
  --model-id acme-support-voice-2026-06-11 \
  --pricing-profile-id acme-cascade-2026-06-11 \
  --pricing-snapshot-date 2026-06-11 \
  --pipeline-type cascaded \
  --trials 1 \
  --output data/openvoicecs/reports/my_agent.json

python scripts/run_openvoicecs.py submission-card data/openvoicecs/reports/my_agent.json \
  --submitter "Example Team" \
  --organization "Example Org" \
  --training-data-statement "No benchmark scenarios used for model training." \
  --safety-statement "System was evaluated against internal support safety checks." \
  --limitations "Prototype adapter; not a production deployment." \
  --output data/openvoicecs/reports/my_agent_submission_card.json

python scripts/run_openvoicecs.py validate-submission-card \
  data/openvoicecs/reports/my_agent_submission_card.json
```

`init-submission` writes a starter adapter with the required `run(scenario,
trial_index)` callable plus slots for messages, tool calls, events, latency,
usage, and cost telemetry. Replace the stub internals with your agent call,
then submit it as `path/to/adapter.py:run`.
Submission cards provide machine-readable model/provider, pricing, modality,
report-hash, reproducibility, and disclosure metadata for leaderboard intake.

Register external-system evidence:

```bash
python scripts/run_openvoicecs.py validate-external-systems \
  --registry data/openvoicecs/external_systems_v0.1.json
# or
make openvoicecs-validate-external-systems
```

The external-system registry is a versioned release artifact that records
reference fixtures, pending external submissions, and official leaderboard
systems separately. Each registered system can pin its report, submission card,
run manifest, release bundle, and judge annotation package by SHA-256. Official
systems must not use the reference provider, must include pricing evidence, must
include a submission card, and must provide non-fixture judge evidence. The v0.1
registry contains only OpenVoiceCS reference fixtures; it intentionally reports
zero official external systems until real third-party runs are collected.

Audio-mode submissions use the same adapter protocol and receive `audio_variant` metadata:

```bash
python scripts/run_openvoicecs.py submit examples/openvoicecs_submission_adapter.py:run \
  --mode audio \
  --track adversarial_compliance \
  --name example_audio_submission
```

Validate saved reports before leaderboard or frontier ingestion:

```bash
python scripts/run_openvoicecs.py validate-report data/openvoicecs/reports/oracle.json
# or
make openvoicecs-validate-reports
```

The report validator checks required top-level fields, score ranges, metric
coverage, result/trial consistency, nonnegative latency/cost telemetry, and
recomputes top-level pass rates, metric averages, and weighted overall score
from the trial records.
The leaderboard, frontier, and release-bundle commands run this validation at
ingestion time and reject invalid saved reports before publishing derived
artifacts.

Validate public/sealed split assignments:

```bash
python scripts/run_openvoicecs.py validate-splits \
  --splits data/openvoicecs/splits_v0.1.json
# or
make openvoicecs-validate-splits
```

The v0.1 suite has both `public_dev` and `sealed_test` assignments. Public IDs
are intended for local debugging and method review. Sealed IDs are committed by
hash and reserved for hosted evaluation and final leaderboard claims.

Build and validate cryptographic split commitments:

```bash
python scripts/run_openvoicecs.py split-commitments \
  --splits data/openvoicecs/splits_v0.1.json \
  --output data/openvoicecs/split_commitments_v0.1.json

python scripts/run_openvoicecs.py validate-split-commitments \
  data/openvoicecs/split_commitments_v0.1.json \
  --splits data/openvoicecs/splits_v0.1.json
```

The commitment file hashes canonical scenario and audio-variant payloads per
split and records a root hash. Public-dev IDs are revealed by default, while
sealed-test IDs are hidden unless `--reveal-sealed-ids` is explicitly used. This
lets a release publish non-public test-set counts and immutable content
commitments before hosted evaluation without exposing sealed prompts or oracles.

Validate sealed-test operations:

```bash
python scripts/run_openvoicecs.py validate-sealed-ops \
  --sealed-ops data/openvoicecs/sealed_ops_v0.1.json \
  --splits data/openvoicecs/splits_v0.1.json \
  --split-commitments data/openvoicecs/split_commitments_v0.1.json
# or
make openvoicecs-validate-sealed-ops
```

The sealed-ops manifest defines custody, permitted roles, conflict
attestation, pre-submission access prohibition, submission-attempt limits,
fixed-environment requirements, audit-log requirements, result-release timing,
and disclosure rules. Validation also checks that the sealed split is non-empty
and that the split-commitment file does not reveal sealed-test IDs.

Validate the hosted sealed-evaluator queue:

```bash
python scripts/run_openvoicecs.py validate-sealed-queue \
  --queue data/openvoicecs/sealed_evaluator_queue_v0.1.json \
  --sealed-ops data/openvoicecs/sealed_ops_v0.1.json \
  --split-commitments data/openvoicecs/split_commitments_v0.1.json
# or
make openvoicecs-validate-sealed-queue
```

The queue manifest is the operational ledger for submissions that pass through
the hosted evaluator. It links each queue entry to a submission intake package,
records attempt limits and attempt artifacts, and requires each attempt to be
served by the hosted evaluator without revealing sealed IDs, raw prompts, or
expected states to submitters. The v0.1 entry is a reference fixture that
exercises the format; official submissions must use real external systems and
non-fixture judging.

Validate provenance, consent, license, and contamination metadata:

```bash
python scripts/run_openvoicecs.py validate-provenance \
  --provenance data/openvoicecs/provenance_v0.1.json
# or
make openvoicecs-validate-provenance
```

The provenance manifest covers every scenario and audio variant. It records
source type, license, authoring method, speaker consent or synthetic voice
status, whether real customer data is present, and contamination risk. Release
audits include provenance coverage and fail release gates when any scenario or
audio variant is missing provenance metadata.

Validate physical audio assets for release candidates:

```bash
python scripts/run_openvoicecs.py pin-audio-assets \
  --audio-manifest data/openvoicecs/audio_manifest_v0.1.json \
  --audio-root . \
  --output data/openvoicecs/audio_manifest_pinned.json

python scripts/run_openvoicecs.py validate-audio-assets \
  --audio-manifest data/openvoicecs/audio_manifest_pinned.json \
  --audio-root .
```

`pin-audio-assets` computes SHA-256, WAV sample rate, and WAV duration from the
collected files and writes a pinned manifest. `validate-audio-assets` then
requires every referenced audio file to exist, every `audio.sha256` field to be
pinned, every hash to match the file bytes, and WAV sample rate and duration
metadata to match the manifest. The v0.1 seed manifest includes pinned
synthetic TTS WAV assets; public beta should add broader consented speaker
coverage.

Validate the subjective-quality judge rubric:

```bash
python scripts/run_openvoicecs.py validate-judge-rubric \
  --rubric data/openvoicecs/judge_rubric_v0.1.json
# or
make openvoicecs-validate-judge-rubric
```

Validate the judging protocol:

```bash
python scripts/run_openvoicecs.py validate-judge-protocol \
  --protocol data/openvoicecs/judging/judge_protocol_v0.1.json
# or
make openvoicecs-validate-judge-protocol
```

The judge protocol is a machine-readable release artifact. It pins the rubric
path, judge prompt path, annotation mode, minimum raters per item, minimum
inter-rater agreement, blinding rules, adjudication trigger, rater
qualification requirements, quality-control fractions, and artifacts that must
be published for judged releases. The v0.1 reference bundle declares
`annotation_mode: reference_fixture` because the included oracle annotations are
deterministic fixtures for harness validation. Official external leaderboard
judging should use the same protocol with trained human raters, audited model
judges, or both.

Validate the judge study:

```bash
python scripts/run_openvoicecs.py validate-judge-study \
  --study data/openvoicecs/judging/judge_study_v0.1.json
# or
make openvoicecs-validate-judge-study
```

The judge study is the study-level governance artifact for official subjective
quality evidence. It binds protocol, rubric, prompt, and annotation package
hashes, then specifies sampling coverage, stratification, rater pool
qualification, calibration items, duplicate/gold controls, blinding,
adjudication, and audit retention. The v0.1 study is a reference fixture and is
not eligible for official judging; official studies must use trained human
raters, audited model judges, or both, and cannot use reference-fixture raters.

Validate the judge annotation package:

```bash
python scripts/run_openvoicecs.py validate-judge-annotation-package \
  --package data/openvoicecs/judging/judge_annotation_package_v0.1.json
# or
make openvoicecs-validate-judge-annotation-package
```

The annotation package is the evidence envelope for judged releases. It
hash-pins the protocol, rubric, prompt, source reports, raw annotation files,
aggregated judge reports, rater manifest, blinding controls, and adjudication
summary. The v0.1 package is marked `official_judging: false` and
`annotation_mode: reference_fixture`; it proves the harness and release plumbing
but does not replace independent human or audited model-judge evidence for
external systems.

Build a judge report from human or model-judge annotations:

```bash
python scripts/run_openvoicecs.py judge-report \
  data/openvoicecs/reports/candidate.json \
  data/openvoicecs/judging/candidate_annotations.jsonl \
  --rubric data/openvoicecs/judge_rubric_v0.1.json \
  --output data/openvoicecs/judging/candidate_judge_report.json

python scripts/run_openvoicecs.py validate-judge-report \
  data/openvoicecs/judging/candidate_judge_report.json
```

Attach a judge report to the original benchmark report when you want frontier
experience gates to use judged quality instead of the deterministic proxy:

```bash
python scripts/run_openvoicecs.py apply-judge-report \
  data/openvoicecs/reports/candidate.json \
  data/openvoicecs/judging/candidate_judge_report.json \
  --output data/openvoicecs/reports/candidate_judged.json
```

Judge reports are separate from deterministic oracle scores. They aggregate
empathy, clarity, naturalness, professionalism, resolution communication, and
voice-channel fit on a 1-5 scale, report item/rater coverage, and compute
Krippendorff's alpha for inter-rater agreement.
`validate-judge-report` enforces the aggregated report contract, minimum raters
per item, score bounds, dimension consistency, paired-rating availability, and
the rubric's minimum release agreement threshold.
Applied reports preserve the deterministic task score and add
`conversation_experience_score`, per-trial `experience_judgment`, coverage,
rater counts, and agreement metadata.

The seed frontier bundle uses deterministic reference annotations for the
oracle reports:

```bash
python scripts/generate_openvoicecs_judge_annotations.py \
  data/openvoicecs/baselines/oracle_text.json \
  --output data/openvoicecs/judging/oracle_text_annotations.jsonl
```

These reference annotations are a reproducible fixture for the seed oracle
bundle. Public judged releases should publish the stated human or model-judge
annotation protocol and keep the deterministic task score separate from judged
experience.

Compare two saved reports statistically:

```bash
python scripts/run_openvoicecs.py compare \
  data/openvoicecs/reports/baseline.json \
  data/openvoicecs/reports/candidate.json \
  --iterations 10000 \
  --seed 0 \
  --output data/openvoicecs/reports/candidate_vs_baseline.json
```

The comparison aligns reports by scenario ID, reports candidate-minus-baseline
deltas, computes paired bootstrap confidence intervals over scenario-level
scores and metric scores, and runs a two-sided exact McNemar test over
discordant `pass^k` outcomes. It also reports stratified paired bootstrap
intervals by domain, track, and difficulty, plus per-slice deltas. Public claims
should inspect these slices before declaring a model better overall; a higher
aggregate score can still hide regressions in regulated, adversarial, or audio
tracks.

Validate leaderboard claims:

```bash
python scripts/run_openvoicecs.py validate-claims \
  --claims data/openvoicecs/claims/leaderboard_claims_v0.1.json
# or
make openvoicecs-validate-claims
```

The claim package is the governance layer for public statements. It pins
comparison reports, baseline and candidate reports, release bundles, and
external-system registry evidence. Official improvement claims must have a
candidate-higher confidence interval, McNemar p-value within policy, no
protected metric regression on privacy/auth/safety, and non-fixture judging
evidence. The v0.1 package contains only a reference sanity claim comparing the
no-op text baseline to the oracle text baseline; it intentionally contains zero
official third-party claims.

Validate the official-submission intake envelope:

```bash
python scripts/run_openvoicecs.py validate-submission-intake \
  --intake data/openvoicecs/submissions/reference_submission_intake_v0.1.json
# or
make openvoicecs-validate-submission-intake
```

The intake envelope is the handoff format for leaderboard submissions. It binds
the submission card, judged report, frozen run manifest, release bundle,
external-system registry, judge annotation package, and leaderboard claim
package with SHA-256 and byte counts. The reference envelope is intentionally
marked `reference_fixture`; official submissions must be marked `official`,
must include the same artifact set, and cannot rely on fixture-generated
judging evidence.

Build a leaderboard from saved reports:

```bash
python scripts/run_openvoicecs.py leaderboard 'data/openvoicecs/reports/*.json' --output data/openvoicecs/leaderboard.json
```

Build a colocated frontier release bundle:

```bash
python scripts/run_openvoicecs.py release-bundle \
  data/openvoicecs/baselines/oracle_text_judged.json \
  data/openvoicecs/baselines/oracle_audio_manifest_judged.json \
  --output-dir data/openvoicecs/releases/frontier_seed \
  --pricing-manifest data/openvoicecs/pricing_snapshot_v0.1.json \
  --changelog data/openvoicecs/changelog_v0.1.json \
  --baseline-manifest data/openvoicecs/baselines/reference_baselines_v0.1.json \
  --review-manifest data/openvoicecs/scenario_reviews_v0.1.json \
  --judge-model openvoicecs-reference-judge-v0.1 \
  --judge-prompt data/openvoicecs/judging/judge_prompt_v0.1.md \
  --pricing-snapshot-date 2026-06-11 \
  --region local \
  --network loopback \
  --hardware-profile local-macos-arm64 \
  --transport in_process \
  --concurrency-levels 1 10 100 \
  --latency-target-ms 300 \
  --cost-target-usd 0.10 \
  --readiness-profile seed
```

The bundle command writes `frontier_report.json`, `run_manifest.json`,
`readiness.json`, `release_bundle.json`, the `plots/` directory, and the
`scorecards/` directory together.
`release_bundle.json` records SHA-256 hashes and validation status for the
generated artifacts. It also copies source inputs into `inputs/` and pins their
hashes, including submitted reports, scenarios, audio/pricing/split/provenance
manifests, release changelog, reference baseline manifest, and optional judge
prompts. A published frontier
release is therefore a self-contained, auditable tuple rather than a set of
loose files. Generated artifacts and snapshotted inputs are recorded with paths
relative to the bundle root. The bundled `run_manifest.json` also points at
those snapshotted `inputs/` files, so validation does not depend on the
original submission locations. Original input locations remain in `source_path`
for audit context.

Verify a published bundle later:

```bash
python scripts/run_openvoicecs.py validate-release-bundle \
  data/openvoicecs/releases/frontier_seed/release_bundle.json
```

The validator checks the bundle contract, source-input hashes, artifact
existence, SHA-256 hashes, byte sizes, frontier JSON, run manifest JSON,
readiness JSON, plot data, SVG plot files, and scorecard JSON/CSV/Markdown
files.

Build a latency-cost-quality frontier report:

```bash
python scripts/run_openvoicecs.py frontier 'data/openvoicecs/reports/*.json' \
  --pricing-manifest data/openvoicecs/pricing_snapshot_v0.1.json \
  --pricing-snapshot-date 2026-06-11 \
  --region us-east-test \
  --network controlled-wired \
  --hardware-profile c7g.2xlarge-client \
  --transport websocket \
  --concurrency-levels 1 10 100 \
  --latency-target-ms 300 \
  --cost-target-usd 0.10 \
  --output data/openvoicecs/frontier/frontier_2026-06-11.json \
  --plot-dir data/openvoicecs/frontier \
  --scorecard-dir data/openvoicecs/frontier/scorecards
```

Validate a saved frontier artifact before publishing or passing it to readiness:

```bash
python scripts/run_openvoicecs.py validate-frontier \
  data/openvoicecs/frontier/frontier_2026-06-11.json
```

The frontier report treats quality, p95 voice-to-voice time-to-first-byte, and cost per successful conversation as separate axes. Systems below the experience gate, missing a primary axis, or missing provider pricing snapshot dates are shown but excluded from the Pareto frontier. Scorecards also include `cost_provenance`, which records direct cost samples, component-derived samples, fully loaded samples, missing cost samples, required components, and per-component coverage counts.

The report also includes `domain_frontiers`, a first-class validated mapping
from each scenario domain to its 3D Pareto frontier and 2D projection frontiers.
These domain frontiers are recomputed from domain-specific latency, cost, and
task-success axes, not copied from the overall frontier.

Latency and cost targets are optional. When provided, the report adds
`constrained_frontiers`, which filters systems to those inside each requested
budget before recomputing the Pareto set.

`validate-frontier` recomputes the overall Pareto frontier, both 2D projection
frontiers, constrained frontiers, and domain frontiers from the saved system
axes. A report with stale or hand-edited frontier names fails validation even if
all referenced system names exist.

An optional user-weighted utility view can be requested with
`--utility quality=1,latency=0.2,cost=0.1`. This is a secondary view and not a
fixed leaderboard. When present, validation recomputes the utility scores and
ranking from the saved system axes and declared weights, so a stale utility
ranking cannot be published as a fixed benchmark order.

Reports may include judged conversation-experience scores. If present,
`conversation_experience_score` is used for the frontier experience gate; if it
is absent, the deterministic `experience_proxy` metric is used as a fallback.
Per-trial judged output can be attached by adapters or offline judge pipelines:

```json
{
  "experience_judgment": {
    "score": 0.82,
    "judge": {
      "type": "llm",
      "model": "claude-opus-4-6",
      "prompt_version": "openvoicecs-exp-v1"
    },
    "dimensions": {
      "naturalness": {"score": 0.8, "note": "clear and conversational"},
      "helpfulness": {"score": 0.9}
    }
  }
}
```

Dimension scores may be reported on a 0-1, 1-5, or 1-10 scale; the harness
normalizes them to 0-1 and records coverage plus judge counts in
`conversation_experience`.

Build the frozen run manifest for a frontier release:

```bash
python scripts/run_openvoicecs.py run-manifest 'data/openvoicecs/reports/*.json' \
  --pricing-manifest data/openvoicecs/pricing_snapshot_v0.1.json \
  --changelog data/openvoicecs/changelog_v0.1.json \
  --baseline-manifest data/openvoicecs/baselines/reference_baselines_v0.1.json \
  --judge-model claude-opus-4-6 \
  --judge-prompt data/openvoicecs/judging/judge_prompt_v0.1.md \
  --seed 0 \
  --region us-east-test \
  --network controlled-wired \
  --hardware-profile c7g.2xlarge-client \
  --transport websocket \
  --concurrency-levels 1 10 100 \
  --output data/openvoicecs/run_manifest.json
```

The run manifest pins the release tuple: scenario suite hash, audio manifest
hash, split manifest hash, pricing manifest hash, changelog hash, reference
baseline manifest hash, report hashes, model metadata, judge model/prompt hash,
seed, region, network, client hardware profile, transport, and concurrency
levels. Validate a saved manifest with:

```bash
python scripts/run_openvoicecs.py validate-run-manifest data/openvoicecs/run_manifest.json
```

Release-ready manifests must use explicit controlled values for region, network,
client hardware profile, transport, and non-empty positive concurrency levels;
placeholder `unspecified` values fail official readiness. When validating a
saved manifest file, the validator also resolves referenced report and manifest
paths, then checks their byte sizes and SHA-256 hashes against the frozen
entries.

Each report in a release should pin system identity and pricing metadata:

```json
{
  "model_metadata": {
    "display_name": "my-agent",
    "provider": "acme",
    "model_id": "voice-agent-2026-06-11",
    "pricing_profile_id": "acme-cascade-2026-06-11",
    "pricing_snapshot_date": "2026-06-11"
  }
}
```

Adapter submissions may use `submission_spec` instead of `model_id`, but still
need provider, pricing snapshot, pricing source, and pipeline type information
in the resolved manifest.

Reports can either embed `model_metadata.pricing` directly or set
`model_metadata.pricing_profile_id` to a profile from the pinned pricing
manifest. The frontier command resolves cascaded ASR+LLM+TTS profiles or native
speech-to-speech profiles before computing cost per successful conversation.

Run the reference realtime/load harness:

```bash
python scripts/run_openvoicecs.py load \
  --endpoint ws://127.0.0.1:8765/openvoicecs \
  --transport websocket \
  --max 50 \
  --trials 3 \
  --concurrency-levels 1 10 100 \
  --region us-east-test \
  --network controlled-wired \
  --hardware-profile c7g.2xlarge-client \
  --pricing-snapshot-date 2026-06-11 \
  --output data/openvoicecs/reports/my_agent_load.json
```

For WebRTC, point `--endpoint` at the signaling endpoint and use the same event
contract over the `openvoicecs` data channel:

```bash
python scripts/run_openvoicecs.py load \
  --endpoint https://127.0.0.1:8765/openvoicecs/webrtc-offer \
  --transport webrtc \
  --max 50 \
  --trials 3 \
  --concurrency-levels 1 10 100 \
  --region us-east-test \
  --network controlled-wired \
  --hardware-profile c7g.2xlarge-client \
  --pricing-snapshot-date 2026-06-11 \
  --output data/openvoicecs/reports/my_agent_webrtc_load.json
```

For local harness checks, omit `--endpoint` and use a built-in adapter:

```bash
python scripts/run_openvoicecs.py load --agent oracle --max 6 --concurrency-levels 1 10 100
```

Built-in realtime adapters emit reference-zero usage and cost telemetry so local
dry-run reports can exercise the frontier pipeline. Official submissions should
replace that with provider usage plus a pinned pricing profile.

Run the minimal custom-agent adapter:

```bash
python examples/openvoicecs_custom_agent.py
```

## Latency-Cost-Quality Frontier

The frontier report is the "MLPerf for Voice Agents" view of the benchmark. It
does not choose a single winner. It normalizes each submitted report into three
primary axes and identifies Pareto non-dominated systems:

| Axis | Primary field | Direction | Definition |
| --- | --- | --- | --- |
| Latency | `p95_v2v_ttfb_ms` | Lower is better | User end-of-speech/VAD endpoint to first audio byte out. |
| Cost | `cost_usd_per_successful_conversation` | Lower is better | Fully loaded conversation cost divided by task success rate. |
| Quality | `task_success_rate` | Higher is better | Task completion/resolution rate on the fixed scenario set. |

Each system also gets a scorecard with p50/p90/p95/p99 TTFB, last-byte latency,
barge-in stop latency, interruption recovery latency, stage attribution
(`asr_finalization_ms`, `llm_ttft_ms`, `tts_first_chunk_ms`), cost per
conversation, cost per successful conversation, task success, experience score,
pricing snapshot date, and `latency_at_100_concurrency_p95_ms` when available.
Scorecards also include `latency_measurement`, which counts how many samples
came from canonical realtime event streams versus precomputed reported latency
or harness runtime fallback. Public and leaderboard readiness require every
latency sample to be backed by `user.end_speech` origin evidence plus
`tts.first_audio` and `agent.complete` events.
Scorecards also include `latency_load`, which records sample counts, saturation,
failed-call counts, peak active calls, and p95 TTFB for each requested
concurrency level. Official readiness requires sample counts of at least 1, 10,
and 100 respectively for the standard load tuple, no failed calls at those
levels, explicit requested/completed/error/peak-active call counts, and
evidence that the requested concurrency was saturated, so a single request
cannot masquerade as a 100-concurrent-call result.

Scorecards include `axis_confidence_intervals` for the primary frontier axes.
Latency and cost intervals use a fixed-seed percentile bootstrap over trial
samples; task-success intervals use source report intervals when present or a
Wilson interval over per-trial pass/fail samples. Frontier validation rejects
missing interval methods, invalid confidence levels, negative bounds, and
intervals where the estimate falls outside `[low, high]`.
Official release profiles enforce repeated trials in both the frozen run
manifest and the scorecard interval sample counts: `public_beta` requires at
least 3 trials per scenario, and `leaderboard_v1` requires at least 5, so
confidence intervals are not based on single samples.
The CLI can export these standardized scorecards as `scorecards.json`,
`scorecards.csv`, and `scorecards.md`; release bundles include those files by
default.

The report includes 2D projection frontiers (`latency_vs_quality` and
`cost_vs_quality`) in addition to the 3D frontier, both overall and per domain.
With `--plot-dir`, the CLI writes `frontier_plot_data.json` plus
dependency-free SVG plots for the overall frontier and for each available
scenario domain. Conversation experience is a gate, not an optimization axis,
so a system cannot appear on the frontier by being cheap or fast while
producing unusable conversations.

Optional constrained frontiers answer budget questions such as
`p95_v2v_ttfb_ms <= 300` and
`cost_usd_per_successful_conversation <= 0.10` without creating a fixed global
leaderboard.

When judged `conversation_experience_score` is available it takes precedence
over the deterministic proxy. Scorecards include `experience_score_source` so
users can tell whether the gate used judged or proxy experience, plus
`experience_evidence` with judged-trial coverage and judge counts. Public and
leaderboard readiness require judged coverage for every scored trial.

## Reference Realtime Client

The realtime load harness sends every system the same `openvoicecs.realtime.v1`
request. The latency origin is always the `user.end_speech` event at `t_ms=0`,
which represents the VAD endpoint. Implementations may be cascaded
ASR+LLM+TTS pipelines or native speech-to-speech models; both are expected to
return the same conversation-level trace.

Canonical request fields:

```json
{
  "protocol": "openvoicecs.realtime.v1",
  "scenario_id": "retail-refund-damaged-item-001",
  "trial_index": 0,
  "concurrency": 10,
  "seed": 0,
  "transcript": "customer: ...",
  "audio": {"path": "data/openvoicecs/audio/...", "format": "wav"},
  "events": [
    {"type": "session.start", "t_ms": 0.0},
    {"type": "user.end_speech", "t_ms": 0.0}
  ]
}
```

WebSocket systems should emit JSON events until `agent.complete`. WebRTC systems
use an SDP signaling endpoint, then emit the same JSON events over the
`openvoicecs` data channel. The reference client recognizes:

| Event | Purpose |
| --- | --- |
| `asr.final` | ASR finalization timestamp. |
| `llm.first_token` | LLM time-to-first-token timestamp. |
| `tts.first_audio` | First audio byte/chunk timestamp, used for v2v TTFB. |
| `barge_in.stop` | Time until the agent stops speaking after barge-in. |
| `barge_in.recovered` | Interruption recovery timestamp. |
| `agent.message` | Agent text transcript for scoring/debugging. |
| `tool.call` | Tool call emitted by the agent. |
| `policy.event` | SOP/auth/privacy event emitted by the agent. |
| `usage` / `cost` | Usage and cost metadata. |
| `agent.complete` | Last audio byte timestamp plus final trace fields. |

The load report records p50/p90/p95/p99 TTFB, last-byte latency, barge-in stop
latency, interruption recovery latency, and per-concurrency summaries. It is
also a normal benchmark report, so it can be passed directly to the frontier
command. Candidate call failures are recorded as failed zero-score trials
instead of aborting the load run; per-concurrency summaries include requested,
completed, failed, and peak-active call counts so overload collapse is visible.
When events are used, the normalized trace stores latency measurement
provenance showing `source=event_stream`, `origin_event=user.end_speech`, and
the first/last audio event names. This provenance is aggregated into frontier
scorecards and checked by official readiness profiles.

## Pricing Snapshot Manifest

Cost comparisons are reproducible only when provider pricing is frozen for a
benchmark release. The default manifest is:

```text
data/openvoicecs/pricing_snapshot_v0.1.json
```

Each manifest has a `snapshot_date`, `currency`, component `entries`, and
pipeline `profiles`. A report may select a profile with:

```json
{
  "model_metadata": {
    "display_name": "my-agent",
    "pricing_profile_id": "reference-zero-v0.1"
  }
}
```

Profiles declare a `pipeline_type`. Cascaded profiles use the default
`cascaded` type and must cover ASR, LLM, TTS, telephony, and transport. Native
speech-to-speech profiles use `native_speech_to_speech` and must cover
speech-to-speech, telephony, and transport. Pricing validation also checks that
each profile component points to an entry with the same declared `component`,
and that each component entry contains at least one numeric rate key that can
actually price that component, so an ASR entry with only LLM token rates is
rejected.

| Component | Example pricing keys |
| --- | --- |
| ASR | `asr_per_minute`, `stt_per_hour` |
| LLM | `input_per_mtok`, `output_per_mtok`, `cached_input_per_mtok` |
| TTS | `tts_per_1k_characters`, `tts_per_minute` |
| Speech-to-speech | `speech_to_speech_per_minute`, `input_audio_per_minute`, `output_audio_per_minute`, `input_audio_per_mtok`, `output_audio_per_mtok` |
| Telephony | `telephony_per_minute`, `phone_per_minute` |
| Transport | `transport_per_minute`, `webrtc_per_minute` |

Native adapters can report `speech_to_speech_seconds`, `input_audio_seconds`,
`output_audio_seconds`, `input_audio_tokens`, and `output_audio_tokens` in
trial `usage`; cascaded adapters can continue reporting ASR seconds, LLM tokens,
and TTS characters or seconds.

Official releases should replace the local zero-cost reference profile with
provider-specific entries and preserve the pricing file hash in the release
audit artifact. Reference profiles are valid for seed/local runs, but they do
not count toward public or leaderboard comparable-pricing coverage; official
profiles need non-reference providers and model IDs. Public and leaderboard
readiness profiles require every scored cost sample to be component-derived
across the required pipeline components; opaque `cost_usd` samples are accepted
for local runs but do not satisfy the fully loaded release evidence gate unless
the same row also includes complete usage and pricing coverage.

## Public And Sealed Splits

Public-dev and sealed-test assignments are versioned separately from scenario
content:

```text
data/openvoicecs/splits_v0.1.json
```

Each scenario and audio variant may appear in exactly one split. The validator
rejects unknown IDs, duplicate IDs within a split, and overlap across splits.
Release audits include split coverage and fail release gates when any scenario
or audio variant is unassigned.

The current v0.1 suite assigns 80 scenarios and 40 audio variants to
`public_dev`, and 121 scenarios and 80 audio variants to `sealed_test`.
`data/openvoicecs/split_commitments_v0.1.json` publishes the commitment root
while hiding sealed IDs. `data/openvoicecs/sealed_ops_v0.1.json` defines the
hosted evaluation custody and disclosure policy. The repository still includes
local fixture data for development and verifier testing; official leaderboard
claims should serve sealed items through the private evaluator described by the
sealed-ops policy.

Release audits also summarize physical audio asset coverage:
existing files, pinned hashes, hash matches, sample-rate matches, duration
matches, positive-duration files, and total decoded duration. The `seed`
readiness profile permits pinned synthetic TTS audio entries; `public_beta` and
`leaderboard_v1` require every audio variant to have a verified, positive-duration
asset.

## Provenance And Contamination

Every release should include:

```text
data/openvoicecs/provenance_v0.1.json
```

Scenario provenance records source type, license, authoring method, real-data
status, and contamination risk. Audio provenance records source type, license,
speaker consent or synthetic voice status, voice rights, real-data status, and
contamination risk. `public_beta` and `leaderboard_v1` readiness require full
provenance coverage, open licenses, no real customer data, low or no
contamination risk, and consent-covered or synthetic audio.

## Release Changelog And Errata

Every release should include a machine-readable changelog:

```text
data/openvoicecs/changelog_v0.1.json
```

Validate it against the scenario suite and audio manifest:

```bash
python scripts/run_openvoicecs.py validate-changelog \
  --changelog data/openvoicecs/changelog_v0.1.json
# or
make openvoicecs-validate-changelog
```

Changelog entries record the release/change ID, change type, date, summary,
compatibility class, affected scenario IDs, affected audio-variant IDs, and
reviewers. Errata entries record known benchmark issues with status and
severity. Release audits include changelog validation, file hash, change
coverage, entry-type counts, compatibility counts, and open-errata counts. A
published scenario ID should never be silently edited; any scoring-affecting
change should be recorded here and treated as a new benchmark release.

## Subjective Judge Rubric

Some voice-agent qualities are hard to judge with deterministic state oracles.
OpenVoiceCS keeps those scores separate and auditable through a versioned rubric:

```text
data/openvoicecs/judge_rubric_v0.1.json
```

Annotations can be JSONL or JSON and must use this shape:

```json
{
  "item_id": "retail-refund-damaged-item-001",
  "scenario_id": "retail-refund-damaged-item-001",
  "rater_id": "human-rater-01",
  "scores": {
    "empathy": 4,
    "clarity": 5,
    "naturalness": 4,
    "professionalism": 5,
    "resolution_communication": 5,
    "channel_fit": 4
  },
  "notes": "Optional short rationale."
}
```

Official judged releases should publish the rubric hash, judge protocol hash,
judge prompt hash, anonymized annotation files when licensing allows it, item
coverage, rater count, adjudication summary, and agreement statistics.
Readiness enforces judged conversation-experience scorecards plus judge
model/prompt pinning for public and leaderboard frontier profiles. A subjective
report should not replace the deterministic task score; it is a separate
scorecard dimension.

The judge protocol is versioned at:

```text
data/openvoicecs/judging/judge_protocol_v0.1.json
```

It requires at least two raters per item, blinded and shuffled item order,
hidden expected actions, adjudication for large disagreements, rater
qualification controls, gold and duplicate quality-control items, and an audit
log. The included v0.1 oracle annotations are deterministic reference fixtures,
not a substitute for independent human-rater evidence on external submissions.

## Scientific Governance Rules

To become a serious open-source benchmark, the project should follow these rules:

- Keep public dev scenarios separate from sealed test scenarios.
- Version every scenario release.
- Require scenario IDs to remain stable after publication.
- Publish deterministic scoring code and per-scenario evidence.
- Report confidence intervals when evaluating stochastic agents.
- Use paired report comparison before claiming one system is materially better than another.
- Report cost, latency, retry count, and tool-call count alongside quality.
- Maintain a scenario bug-bounty process and changelog.
- Publish provenance manifests with source, license, consent, and contamination metadata.
- Require human audit samples for subjective experience scoring before official judged releases.
- Track contamination risk by avoiding real public support transcripts in sealed test sets unless they are newly collected and licensed.

## Near-Term Roadmap

1. Run real external voice-agent systems through the adapter protocol and publish reproducible submission intake packages.
2. Collect independent multi-rater human or audited model-judge annotations against the published judge protocol.
3. Replace reference-fixture judged oracle annotations with official judged release evidence for external systems.
4. Stand up the hosted sealed evaluator described by `sealed_ops_v0.1.json` and operated through `sealed_evaluator_queue_v0.1.json`.
5. Add broader consented speaker and acoustic-condition coverage beyond the synthetic TTS seed assets.
6. Publish official leaderboard guidance with confidence intervals, paired comparisons, and slice-level regression rules.
