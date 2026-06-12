# OpenVoiceCS-Bench Governance

OpenVoiceCS-Bench is intended to become a scientific, open benchmark for voice
AI customer-service agents. Governance prioritizes reproducibility, benchmark
integrity, contamination control, and clear reporting over rapid leaderboard
growth.

## Release Classes

- `seed`: small public package for method review and local development.
- `public_beta`: expanded public-dev release with meaningful coverage targets,
  provenance coverage, reference baselines, and judged experience samples.
- `leaderboard_v1`: official leaderboard release with frozen run manifests,
  public-dev plus sealed-test policy, statistical comparison guidance, and
  complete release bundles.

Readiness profiles in code are the source of truth for machine-enforced release
criteria.

## Versioning

Scenario suites, audio manifests, split manifests, provenance manifests,
changelogs, datasheets, baseline manifests, scenario review manifests, judge
protocols, judge annotation packages, sealed-test operations manifests, and
release bundles are versioned release artifacts. External-system registries and
leaderboard claim packages are also versioned release
artifacts because they distinguish reference fixtures from official external
leaderboard evidence.
Published IDs are stable. Scoring-affecting changes require a new release
version or a recorded erratum.

Each release must publish:

- scenario suite hash;
- audio manifest hash;
- split manifest and split commitment hashes;
- provenance manifest hash;
- pricing snapshot hash;
- changelog hash;
- reference baseline manifest hash;
- scenario review manifest hash;
- judge protocol hash;
- judge annotation package hash;
- sealed-test operations manifest hash;
- external-system registry hash;
- leaderboard claim package hash;
- datasheet hash;
- release audit result;
- run manifest and bundle hashes for official frontier releases.

## Public And Sealed Splits

Public-dev scenarios exist for development, debugging, and reproducibility.
Sealed-test scenarios must not be exposed through public examples, training
data, prompt templates, logs, or model feedback loops.

Split commitments may publish sealed-test counts and hashes without revealing
sealed item IDs. Any accidental sealed-test disclosure must be recorded as an
erratum and may require rotating the affected sealed set.

Official leaderboard evaluation must follow the sealed-test operations manifest:
no pre-submission sealed access for candidate builders, conflict attestations
for operators and auditors, fixed evaluation environment, bounded submission
attempts, audit logs, and aggregate-first disclosure.

## Scenario Review

Each scenario should be reviewed for:

- replayable tool contract and expected final state;
- SOP and policy realism;
- privacy and authentication coverage;
- forbidden-action coverage;
- grounding probes;
- voice-agent relevance;
- domain and difficulty labels;
- contamination and provenance metadata;
- split assignment.

Release candidates must carry a scenario review manifest showing the required
checklist approvals and minimum reviewer count for every scenario in the suite.

Synthetic scenarios are acceptable when they are realistic and documented.
Real customer data is out of scope unless it is newly collected, consented,
licensed, anonymized, and reviewed for contamination risk.

## Errata

Errata are tracked in the release changelog with status and severity. Critical
errata are issues that can change material conclusions, leak sealed-test
content, compromise privacy/consent claims, or make official scores invalid.

Resolved errata must explain:

- affected scenario/audio IDs;
- whether scores are expected to change;
- compatibility class;
- reviewers;
- replacement release version when applicable.

## Leaderboard Claims

Official claims must cite a frozen run manifest and release bundle. Aggregate
score improvements should be reported with paired comparison statistics and
slice-level inspection across domain, track, difficulty, safety, privacy, and
authentication categories.

External systems must be listed in the external-system registry before they are
used for official claims. Registry entries must distinguish `reference_fixture`,
`pending_external`, and `official` status. Official entries require a
submission card, report, run manifest, release bundle, pricing profile, judge
study, judge annotation package, leaderboard claim package, official submission intake
envelope, sealed-evaluator queue entry, and non-fixture judging evidence.
Judge studies must define sampling, calibration, rater-pool qualification,
blinding, adjudication, and audit controls. Judge annotation packages must pin
the source report, raw annotations, aggregated judge report, rater manifest,
blinding controls, and adjudication summary for each judged system.

Do not claim a single overall winner when a system regresses materially on
regulated, adversarial, privacy, authentication, or audio robustness slices.
Official improvement claims must be listed in the leaderboard claim package and
must include pairwise comparison evidence with confidence intervals, McNemar
statistics, release-bundle evidence, external-system registry evidence, and
non-fixture judging evidence. Official submissions must also pass
`validate-submission-intake`, which binds the submission card, judged report,
run manifest, release bundle, external-system registry, judge annotation
package, and claim package by SHA-256 and byte size.
Official sealed submissions must also pass `validate-sealed-queue`, which
enforces attempt limits, append-only audit-log policy, hosted evaluator serving,
and no sealed ID, raw prompt, or expected-state export to submitters.
Official judged submissions must pass `validate-judge-study` and
`validate-judge-annotation-package`; reference-fixture rater pools are never
eligible for official subjective-quality claims.

## Maintainer Duties

Maintainers are responsible for:

- preserving split boundaries;
- requiring release verification in CI;
- requiring judge-protocol and sealed-ops validation in CI;
- requiring judge-study validation in CI;
- requiring judge-annotation package validation in CI;
- requiring external-system registry validation in CI;
- requiring leaderboard claim package validation in CI;
- reviewing provenance and licensing changes;
- keeping baseline reports reproducible;
- maintaining changelog and errata discipline;
- documenting readiness-profile promotions;
- rejecting changes that overfit the benchmark or weaken scientific claims.
