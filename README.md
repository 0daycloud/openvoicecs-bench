# OpenVoiceCS-Bench

OpenVoiceCS-Bench is an open benchmark for voice AI customer-service agents.
It evaluates whether an agent resolves customer-service tasks, follows SOPs,
uses tools correctly, protects privacy, preserves authentication boundaries,
and reports latency/cost evidence in a reproducible release bundle.

The current repository contains a v0.1 leaderboard-scale release with
deterministic scoring, scenario/audio/provenance/split/changelog/baseline/review
manifests, judge protocol, sealed-test operations policy, submission adapters,
external-system registry, readiness checks, datasheets, and release-bundle tooling.

## Quick Start

Validate the benchmark package:

```bash
python scripts/run_openvoicecs.py verify-release \
  --require-audio-assets \
  --readiness-profile leaderboard_v1 \
  --frontier-report data/openvoicecs/releases/frontier_seed/frontier_report.json \
  --run-manifest data/openvoicecs/releases/frontier_seed/run_manifest.json \
  --plot-dir data/openvoicecs/releases/frontier_seed/plots \
  --strict
```

Run the reference baselines:

```bash
python scripts/run_openvoicecs.py baselines \
  --trials 3 \
  --output-dir data/openvoicecs/baselines \
  --output data/openvoicecs/baselines/reference_baselines_v0.1.json
```

Score the oracle reference agent:

```bash
python scripts/run_openvoicecs.py score --agent oracle --trials 3
```

Score an external adapter:

```bash
python scripts/run_openvoicecs.py init-submission submissions/my_agent.py
python scripts/run_openvoicecs.py submit submissions/my_agent.py:run \
  --name my_agent \
  --provider my_org \
  --model-id my_model \
  --trials 3 \
  --output data/openvoicecs/reports/my_agent.json
```

## Release Artifacts

The v0.1 release is under `data/openvoicecs/` and includes:

- `scenarios_v0.1.json`: deterministic scenario suite.
- `audio_manifest_v0.1.json`: audio/robustness variant manifest.
- `splits_v0.1.json` and `split_commitments_v0.1.json`: public/sealed split metadata and commitments.
- `provenance_v0.1.json`: source, license, consent, real-data, and contamination metadata.
- `changelog_v0.1.json`: release changelog and errata.
- `baselines/reference_baselines_v0.1.json`: reproducible oracle/no-op baseline reports.
- `judging/`: reference judge prompt, annotations, and judged oracle reports for the seed frontier bundle.
- `judging/judge_protocol_v0.1.json`: rater, blinding, adjudication, and quality-control protocol.
- `judging/judge_study_v0.1.json`: human/audited judge study design covering sampling, calibration, rater pool, adjudication, and audit rules.
- `judging/judge_annotation_package_v0.1.json`: hash-pinned annotation, rater, blinding, adjudication, and judge-report evidence.
- `sealed_ops_v0.1.json`: sealed-test custody, access, submission, audit-log, and disclosure policy.
- `sealed_evaluator_queue_v0.1.json`: hosted sealed-evaluator queue with submission attempts, artifact hashes, and exposure controls.
- `external_systems_v0.1.json`: hash-pinned registry of reference fixtures, pending external systems, and official leaderboard evidence.
- `claims/leaderboard_claims_v0.1.json`: claim package for statistical comparisons and official-claim evidence.
- `submissions/reference_submission_intake_v0.1.json`: complete hash-pinned intake envelope tying card, report, run manifest, release bundle, registry, judging, and claims evidence together.
- `scenario_reviews_v0.1.json`: scenario approval and checklist review manifest.
- `datasheet_v0.1.json`: machine-readable datasheet.
- `release_audit.json`: release hashes, validation, stats, and gates.
- `releases/frontier_seed/`: validated frontier report, plots, scorecards, frozen run manifest, and release bundle.

The benchmark documentation is in `docs/openvoicecs_bench.md`.

## Release Gate

Every release candidate must pass:

```bash
make verify-release
make validate-release-bundle
pytest tests/unit/ -q
```

For releases that include physical WAV assets, also run:

```bash
python scripts/run_openvoicecs.py verify-release --require-audio-assets --strict
```

The v0.1 release includes pinned synthetic TTS WAV assets for deterministic
local audio checks. Broader public releases should add more consented speaker
coverage and externally produced baseline runs.

## Contributing

See `CONTRIBUTING.md` for scenario authoring rules, release-file update
requirements, and pull request checks. See `GOVERNANCE.md` for benchmark
versioning, errata, and sealed-test policies.
