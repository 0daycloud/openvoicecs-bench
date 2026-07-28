.PHONY: install dev test lint validate validate-reviews validate-submission-intake \
	audit datasheet verify-release validate-release-bundle baselines \
	submit-example score-provider-help check \
	check-tool-arguments check-forbidden-events validity-gates

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -e ".[dev,providers]"

test:
	$(PYTHON) -m pytest tests/unit -v

lint:
	$(PYTHON) -m ruff check .

validate:
	$(PYTHON) scripts/run_openvoicecs.py validate

validate-reviews:
	$(PYTHON) scripts/run_openvoicecs.py validate-reviews --review-manifest data/openvoicecs/scenario_reviews_v0.1.json

validate-submission-intake:
	$(PYTHON) scripts/run_openvoicecs.py validate-submission-intake --intake data/openvoicecs/submissions/reference_submission_intake_v0.1.json

audit:
	$(PYTHON) scripts/run_openvoicecs.py audit --output data/openvoicecs/release_audit.json

datasheet:
	$(PYTHON) scripts/run_openvoicecs.py datasheet --output data/openvoicecs/datasheet_v0.1.json

verify-release:
	$(PYTHON) scripts/run_openvoicecs.py verify-release --require-audio-assets --readiness-profile leaderboard_v1 --frontier-report data/openvoicecs/releases/frontier_seed/frontier_report.json --run-manifest data/openvoicecs/releases/frontier_seed/run_manifest.json --plot-dir data/openvoicecs/releases/frontier_seed/plots --judge-protocol data/openvoicecs/judging/judge_protocol_v0.1.json --judge-study data/openvoicecs/judging/judge_study_v0.1.json --judge-annotation-package data/openvoicecs/judging/judge_annotation_package_v0.1.json --sealed-ops data/openvoicecs/sealed_ops_v0.1.json --sealed-queue data/openvoicecs/sealed_evaluator_queue_v0.1.json --external-endpoint-contract data/openvoicecs/external_endpoint_contract_v0.1.json --external-systems data/openvoicecs/external_systems_v0.1.json --claims data/openvoicecs/claims/leaderboard_claims_v0.1.json --submission-intake data/openvoicecs/submissions/reference_submission_intake_v0.1.json --strict

validate-release-bundle:
	$(PYTHON) scripts/run_openvoicecs.py validate-release-bundle data/openvoicecs/releases/frontier_seed/release_bundle.json

baselines:
	$(PYTHON) scripts/run_openvoicecs.py baselines --trials 3 --output-dir data/openvoicecs/baselines --output data/openvoicecs/baselines/reference_baselines_v0.1.json

submit-example:
	$(PYTHON) scripts/run_openvoicecs.py submit examples/openvoicecs_submission_adapter.py:run --name example_submission --trials 1

score-provider-help:
	$(PYTHON) scripts/run_openvoicecs.py score-provider --help

# Scenario validity gates. Both scripts exit non-zero when the corpus
# reintroduces a scoring-validity bug, so they are cheap regression guards.
check-tool-arguments:
	$(PYTHON) scripts/mark_ungrounded_tool_arguments.py --check

check-forbidden-events:
	$(PYTHON) scripts/bind_forbidden_event_triggers.py --check

validity-gates: check-tool-arguments check-forbidden-events

# Full pre-release gate: everything CI runs, in one target.
check: lint validity-gates validate validate-reviews validate-submission-intake verify-release validate-release-bundle test
