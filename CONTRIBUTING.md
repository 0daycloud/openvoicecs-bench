# Contributing To OpenVoiceCS-Bench

Contributions should improve benchmark validity, reproducibility, coverage, or
release integrity. Avoid changes that only make a model look better without
improving the benchmark contract.

## Local Checks

Run these before opening a pull request:

```bash
python scripts/run_openvoicecs.py validate-judge-protocol --protocol data/openvoicecs/judging/judge_protocol_v0.1.json
python scripts/run_openvoicecs.py validate-judge-study --study data/openvoicecs/judging/judge_study_v0.1.json
python scripts/run_openvoicecs.py validate-judge-annotation-package --package data/openvoicecs/judging/judge_annotation_package_v0.1.json
python scripts/run_openvoicecs.py validate-sealed-ops --sealed-ops data/openvoicecs/sealed_ops_v0.1.json --splits data/openvoicecs/splits_v0.1.json --split-commitments data/openvoicecs/split_commitments_v0.1.json
python scripts/run_openvoicecs.py validate-sealed-queue --queue data/openvoicecs/sealed_evaluator_queue_v0.1.json --sealed-ops data/openvoicecs/sealed_ops_v0.1.json --split-commitments data/openvoicecs/split_commitments_v0.1.json
python scripts/run_openvoicecs.py validate-external-systems --registry data/openvoicecs/external_systems_v0.1.json
python scripts/run_openvoicecs.py validate-claims --claims data/openvoicecs/claims/leaderboard_claims_v0.1.json
python scripts/run_openvoicecs.py validate-submission-intake --intake data/openvoicecs/submissions/reference_submission_intake_v0.1.json
make openvoicecs-verify-release
make openvoicecs-validate-release-bundle
pytest tests/unit/ -q
```

If you edit Python code, also run:

```bash
python3 -m py_compile \
  src/evaluation/benchmark/*.py \
  scripts/run_openvoicecs.py
```

## Scenario Changes

Do not edit a published scenario silently. Scenario changes must either:

- add new scenario IDs in a new versioned scenario file, or
- fix an erratum and record the change in `data/openvoicecs/changelog_v0.1.json`.

Each new scenario must include:

- stable `id`, `domain`, `track`, `difficulty`, and tags;
- synthetic or consented source provenance;
- an explicit initial state;
- replayable tool definitions;
- deterministic oracle checks for expected state, required/forbidden events,
  privacy, authentication, grounding, and forbidden tool use;
- split assignment in `splits_v*.json`;
- provenance entry in `provenance_v*.json`;
- changelog entry covering the scenario ID.
- scenario review entry with required checklist approvals in `scenario_reviews_v*.json`.

Use the scaffold workflow for batches:

```bash
python scripts/run_openvoicecs.py coverage-plan --profile public_beta
python scripts/run_openvoicecs.py scaffold-scenarios \
  --profile public_beta \
  --count 10 \
  --output drafts/openvoicecs_public_beta_scaffold.json
python scripts/run_openvoicecs.py add-scenarios drafts/new_scenarios.json \
  --output-scenarios data/openvoicecs/scenarios_v0.2.json \
  --output-splits data/openvoicecs/splits_v0.2.json \
  --output-provenance data/openvoicecs/provenance_v0.2.json
```

## Audio Changes

Audio entries must be synthetic or explicitly consented and license-compatible.
For official audio releases, pin file SHA-256, duration, sample rate, and
format:

```bash
python scripts/run_openvoicecs.py pin-audio-assets \
  --audio-manifest data/openvoicecs/audio_manifest_v0.1.json \
  --audio-root . \
  --output data/openvoicecs/audio_manifest_pinned.json
python scripts/run_openvoicecs.py verify-release --require-audio-assets --strict
```

## Baselines And Datasheets

If scenario, audio, scoring, or release metadata changes, regenerate and commit:

```bash
make openvoicecs-baselines
make openvoicecs-validate-reviews
make openvoicecs-audit
make openvoicecs-datasheet
make openvoicecs-validate-submission-intake
make openvoicecs-verify-release
make openvoicecs-validate-release-bundle
```

Baseline reports must be deterministic enough that repeated generation does not
change their hashes. Do not change expected baseline values without explaining
the scoring or scenario change that caused it.

## Pull Request Requirements

A benchmark PR should state:

- what scientific or operational validity gap it addresses;
- which release files changed;
- whether scoring behavior changed;
- whether any public-dev or sealed-test content moved;
- validation commands and results;
- whether judge protocol, judge study, judge annotation package, sealed operations, external-system registry, claim package, or release-bundle artifacts changed;
- contamination, licensing, and consent implications.

Reviewers should block PRs that weaken release gates, remove provenance,
collapse split boundaries, or make unrecorded scoring-affecting changes.
