## Summary

- Scientific/operational validity gap addressed:
- Release files changed:
- Scoring behavior changed: yes/no

## Benchmark Integrity

- Public-dev or sealed-test content moved: yes/no
- Changelog/errata updated: yes/no
- Provenance/license/consent implications:
- Contamination-risk implications:
- Reference baselines regenerated: yes/no/not needed
- Datasheet/audit regenerated: yes/no/not needed
- Judge protocol or sealed-ops policy changed: yes/no
- Judge study changed: yes/no
- Sealed evaluator queue changed: yes/no
- Judge annotation package changed: yes/no
- External-system registry changed: yes/no
- Leaderboard claim package changed: yes/no
- Submission intake package changed: yes/no
- Release bundle regenerated: yes/no/not needed

## Validation

```bash
make check
```

`make check` runs scenario validation, review-manifest validation, submission
intake validation, the strict release gate, release-bundle verification, and
the unit tests.

If you touched judging, sealed-test policy, external systems, or claims, also
run the validators for those artifacts:

```bash
python scripts/run_openvoicecs.py validate-judge-protocol --protocol data/openvoicecs/judging/judge_protocol_v0.1.json
python scripts/run_openvoicecs.py validate-judge-study --study data/openvoicecs/judging/judge_study_v0.1.json
python scripts/run_openvoicecs.py validate-judge-annotation-package --package data/openvoicecs/judging/judge_annotation_package_v0.1.json
python scripts/run_openvoicecs.py validate-sealed-ops --sealed-ops data/openvoicecs/sealed_ops_v0.1.json --splits data/openvoicecs/splits_v0.1.json --split-commitments data/openvoicecs/split_commitments_v0.1.json
python scripts/run_openvoicecs.py validate-sealed-queue --queue data/openvoicecs/sealed_evaluator_queue_v0.1.json --sealed-ops data/openvoicecs/sealed_ops_v0.1.json --split-commitments data/openvoicecs/split_commitments_v0.1.json
python scripts/run_openvoicecs.py validate-external-systems --registry data/openvoicecs/external_systems_v0.1.json
python scripts/run_openvoicecs.py validate-claims --claims data/openvoicecs/claims/leaderboard_claims_v0.1.json
```

Paste relevant output or explain why a command was not run.

## Checklist

- [ ] No API key, credential, or sealed-test content appears in the diff.
- [ ] New Python code passes `ruff check` on the files I touched.
- [ ] Scoring-affecting changes are recorded in the changelog per `GOVERNANCE.md`.
