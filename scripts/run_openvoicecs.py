"""Run OpenVoiceCS-Bench deterministic evaluations.

Examples:
    python scripts/run_openvoicecs.py validate
    python scripts/run_openvoicecs.py score --agent oracle --trials 3
    python scripts/run_openvoicecs.py score --agent noop
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.logging import setup_logging
from src.evaluation.benchmark.baselines import (
    DEFAULT_BASELINE_DIR,
    DEFAULT_BASELINE_MANIFEST_PATH,
    build_reference_baselines,
    validate_reference_baselines_file,
)
from src.evaluation.benchmark.changelog import (
    DEFAULT_CHANGELOG_PATH,
    validate_changelog_file,
)
from src.evaluation.benchmark.claims import (
    DEFAULT_CLAIMS_MANIFEST_PATH,
    validate_claims_manifest_file,
)
from src.evaluation.benchmark.comparison import compare_reports
from src.evaluation.benchmark.coverage import (
    DEFAULT_COVERAGE_TARGET_PATH,
    build_coverage_plan,
)
from src.evaluation.benchmark.datasheet import (
    DEFAULT_DATASHEET_PATH,
    build_benchmark_datasheet_file,
    validate_benchmark_datasheet_file,
)
from src.evaluation.benchmark.external_endpoint import (
    DEFAULT_EXTERNAL_ENDPOINT_CONTRACT_PATH,
    score_external_endpoint,
    validate_external_endpoint_contract_file,
)
from src.evaluation.benchmark.external_systems import (
    DEFAULT_EXTERNAL_SYSTEMS_PATH,
    validate_external_systems_registry_file,
)
from src.evaluation.benchmark.frontier import (
    build_frontier_report,
    validate_frontier_report_file,
    write_frontier_artifacts,
    write_scorecard_artifacts,
)
from src.evaluation.benchmark.judging import (
    DEFAULT_JUDGE_ANNOTATION_PACKAGE_PATH,
    DEFAULT_JUDGE_PROTOCOL_PATH,
    DEFAULT_JUDGE_RUBRIC_PATH,
    DEFAULT_JUDGE_STUDY_PATH,
    apply_judge_report,
    apply_judge_report_from_files,
    build_judge_report,
    build_judge_report_from_files,
    generate_model_judge_annotations,
    load_judge_protocol,
    parse_model_judge_spec,
    validate_judge_annotation_package_file,
    validate_judge_protocol_file,
    validate_judge_report_file,
    validate_judge_rubric_file,
    validate_judge_study_manifest_file,
    write_judge_annotations_jsonl,
)
from src.evaluation.benchmark.openvoicecs import (
    DEFAULT_AUDIO_MANIFEST_PATH,
    DEFAULT_SCENARIO_PATH,
    DEFAULT_SUBMISSION_INTAKE_PATH,
    OpenVoiceCSBench,
    build_leaderboard,
    build_release_audit,
    load_reports,
    no_op_agent,
    oracle_agent,
    pin_audio_manifest_assets_file,
    validate_audio_assets_file,
    validate_audio_manifest_file,
    validate_report,
    validate_report_file,
    validate_suite_file,
)
from src.evaluation.benchmark.pricing import DEFAULT_PRICING_MANIFEST_PATH, load_pricing_manifest
from src.evaluation.benchmark.provenance import (
    DEFAULT_PROVENANCE_MANIFEST_PATH,
    validate_provenance_manifest_file,
)
from src.evaluation.benchmark.provider_adapters import (
    DEFAULT_MODEL_IDS,
    PIPELINE_PROVIDERS,
    build_provider_spec,
    load_workspace_env,
)
from src.evaluation.benchmark.readiness import (
    RELEASE_PROFILES,
    evaluate_release_readiness,
)
from src.evaluation.benchmark.realtime import (
    ReferenceRealtimeClient,
    WebRTCRealtimeClient,
    WebSocketRealtimeClient,
    builtin_realtime_agent,
    run_openvoicecs_realtime_load,
)
from src.evaluation.benchmark.release_bundle import (
    build_frontier_release_bundle,
    validate_frontier_release_bundle_file,
)
from src.evaluation.benchmark.release_verification import (
    DEFAULT_RELEASE_AUDIT_PATH,
    verify_openvoicecs_release,
)
from src.evaluation.benchmark.reviews import (
    DEFAULT_REVIEW_MANIFEST_PATH,
    validate_review_manifest_file,
)
from src.evaluation.benchmark.run_manifest import (
    build_run_manifest,
    validate_run_manifest_file,
)
from src.evaluation.benchmark.scenario_authoring import (
    add_scenarios_to_release_files,
    scaffold_scenario_drafts,
)
from src.evaluation.benchmark.sealed import (
    DEFAULT_SEALED_OPS_PATH,
    DEFAULT_SEALED_QUEUE_PATH,
    validate_sealed_ops_manifest_file,
    validate_sealed_queue_manifest_file,
)
from src.evaluation.benchmark.splits import (
    DEFAULT_SPLIT_COMMITMENT_PATH,
    DEFAULT_SPLIT_MANIFEST_PATH,
    build_split_commitments_file,
    validate_split_commitments_file,
    validate_split_manifest_file,
)
from src.evaluation.benchmark.submission import (
    build_submission_card_from_file,
    score_provider,
    score_submission,
    submission_intake_stats,
    validate_submission_card_file,
    validate_submission_intake_file,
    write_submission_template,
)


def cmd_validate(args: argparse.Namespace) -> None:
    scenario_issues = validate_suite_file(args.scenarios)
    scenario_ids = None
    if not scenario_issues:
        scenario_ids = {
            scenario["id"]
            for scenario in OpenVoiceCSBench.load(args.scenarios).scenarios
        }
    audio_issues = (
        validate_audio_manifest_file(args.audio_manifest, scenario_ids=scenario_ids)
        if args.audio_manifest else []
    )
    issues = scenario_issues + audio_issues
    if issues:
        print("Validation failed:")
        for issue in issues:
            print(f"  {issue.scenario_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)

    bench = OpenVoiceCSBench.load(args.scenarios)
    print(f"Loaded OpenVoiceCS-Bench v{bench.version}: {len(bench.scenarios)} scenarios")
    print("Domains:")
    for domain, count in bench._domain_stats().items():
        print(f"  {domain}: {count}")
    print("Tracks:")
    for track, count in bench._track_stats().items():
        print(f"  {track}: {count}")
    if args.audio_manifest:
        print(f"Audio manifest valid: {args.audio_manifest}")


def cmd_validate_report(args: argparse.Namespace) -> None:
    all_issues = []
    for report_path in args.reports:
        issues = validate_report_file(report_path)
        if issues:
            all_issues.append((report_path, issues))

    if all_issues:
        print("Report validation failed:")
        for report_path, issues in all_issues:
            print(f"  {report_path}:")
            for issue in issues:
                print(f"    {issue.scenario_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)

    print(f"Validated {len(args.reports)} report file(s)")


def cmd_validate_frontier(args: argparse.Namespace) -> None:
    all_issues = []
    for report_path in args.reports:
        issues = validate_frontier_report_file(report_path)
        if issues:
            all_issues.append((report_path, issues))

    if all_issues:
        print("Frontier report validation failed:")
        for report_path, issues in all_issues:
            print(f"  {report_path}:")
            for issue in issues:
                print(f"    {issue.path}: {issue.message}")
        raise SystemExit(1)

    print(f"Validated {len(args.reports)} frontier report file(s)")


def cmd_validate_run_manifest(args: argparse.Namespace) -> None:
    issues = validate_run_manifest_file(args.manifest)
    if issues:
        print("Run manifest validation failed:")
        for issue in issues:
            print(f"  {issue.path}: {issue.message}")
        raise SystemExit(1)
    print(f"Validated run manifest: {args.manifest}")


def cmd_validate_release_bundle(args: argparse.Namespace) -> None:
    all_issues = []
    for bundle_path in args.bundles:
        issues = validate_frontier_release_bundle_file(bundle_path)
        if issues:
            all_issues.append((bundle_path, issues))

    if all_issues:
        print("Release bundle validation failed:")
        for bundle_path, issues in all_issues:
            print(f"  {bundle_path}:")
            for issue in issues:
                print(f"    {issue.path}: {issue.message}")
        raise SystemExit(1)

    print(f"Validated {len(args.bundles)} release bundle file(s)")


def cmd_verify_release(args: argparse.Namespace) -> None:
    verification = verify_openvoicecs_release(
        scenario_path=args.scenarios,
        audio_manifest_path=args.audio_manifest,
        audio_asset_root=args.audio_root,
        require_audio_assets=args.require_audio_assets,
        pricing_manifest_path=args.pricing_manifest,
        split_manifest_path=args.splits,
        split_commitment_path=args.split_commitments,
        provenance_manifest_path=args.provenance,
        changelog_path=args.changelog,
        baseline_manifest_path=args.baseline_manifest,
        review_manifest_path=args.review_manifest,
        datasheet_path=args.datasheet,
        judge_protocol_path=args.judge_protocol,
        judge_study_path=args.judge_study,
        judge_annotation_package_path=args.judge_annotation_package,
        sealed_ops_path=args.sealed_ops,
        sealed_queue_path=args.sealed_queue,
        external_endpoint_contract_path=args.external_endpoint_contract,
        external_systems_path=args.external_systems,
        claims_manifest_path=args.claims,
        submission_intake_path=args.submission_intake,
        release_audit_path=args.release_audit,
        readiness_profile=args.readiness_profile,
        frontier_report_path=args.frontier_report,
        run_manifest_path=args.run_manifest,
        plot_dir=args.plot_dir,
    )
    _print_release_verification(verification)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(verification, f, indent=2)
        print(f"\nSaved release verification to {output}")
    if args.strict and not verification["passed"]:
        raise SystemExit(1)


def cmd_validate_judge_rubric(args: argparse.Namespace) -> None:
    issues = validate_judge_rubric_file(args.rubric)
    if issues:
        print("Judge rubric validation failed:")
        for issue in issues:
            print(f"  {issue.item_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)
    print(f"Validated judge rubric: {args.rubric}")


def cmd_validate_judge_protocol(args: argparse.Namespace) -> None:
    issues = validate_judge_protocol_file(args.protocol)
    if issues:
        print("Judge protocol validation failed:")
        for issue in issues:
            print(f"  {issue.item_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)
    print(f"Validated judge protocol: {args.protocol}")


def cmd_validate_judge_study(args: argparse.Namespace) -> None:
    issues = validate_judge_study_manifest_file(args.study)
    if issues:
        print("Judge study validation failed:")
        for issue in issues:
            print(f"  {issue.item_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)
    print(f"Validated judge study: {args.study}")


def cmd_validate_judge_annotation_package(args: argparse.Namespace) -> None:
    issues = validate_judge_annotation_package_file(args.package)
    if issues:
        print("Judge annotation package validation failed:")
        for issue in issues:
            print(f"  {issue.item_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)
    print(f"Validated judge annotation package: {args.package}")


def cmd_validate_judge_report(args: argparse.Namespace) -> None:
    all_issues = []
    for report_path in args.reports:
        issues = validate_judge_report_file(report_path)
        if issues:
            all_issues.append((report_path, issues))

    if all_issues:
        print("Judge report validation failed:")
        for report_path, issues in all_issues:
            print(f"  {report_path}:")
            for issue in issues:
                print(f"    {issue.item_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)

    print(f"Validated {len(args.reports)} judge report file(s)")


def cmd_validate_sealed_ops(args: argparse.Namespace) -> None:
    issues = validate_sealed_ops_manifest_file(
        args.sealed_ops,
        split_manifest_path=args.splits,
        split_commitment_path=args.split_commitments,
    )
    if issues:
        print("Sealed operations validation failed:")
        for issue in issues:
            print(f"  {issue.item_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)
    print(f"Validated sealed operations: {args.sealed_ops}")


def cmd_validate_sealed_queue(args: argparse.Namespace) -> None:
    issues = validate_sealed_queue_manifest_file(
        args.queue,
        sealed_ops_path=args.sealed_ops,
        split_commitment_path=args.split_commitments,
    )
    if issues:
        print("Sealed evaluator queue validation failed:")
        for issue in issues:
            print(f"  {issue.item_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)
    print(f"Validated sealed evaluator queue: {args.queue}")


def cmd_validate_external_systems(args: argparse.Namespace) -> None:
    issues = validate_external_systems_registry_file(args.registry)
    if issues:
        print("External systems registry validation failed:")
        for issue in issues:
            print(f"  {issue.item_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)
    print(f"Validated external systems registry: {args.registry}")


def cmd_validate_external_endpoint_contract(args: argparse.Namespace) -> None:
    issues = validate_external_endpoint_contract_file(args.contract)
    if issues:
        print("External endpoint contract validation failed:")
        for issue in issues:
            print(f"  {issue.item_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)
    print(f"Validated external endpoint contract: {args.contract}")


def cmd_validate_claims(args: argparse.Namespace) -> None:
    issues = validate_claims_manifest_file(args.claims)
    if issues:
        print("Leaderboard claims validation failed:")
        for issue in issues:
            print(f"  {issue.item_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)
    print(f"Validated leaderboard claims: {args.claims}")


def cmd_validate_splits(args: argparse.Namespace) -> None:
    bench = OpenVoiceCSBench.load(args.scenarios)
    scenario_ids = {scenario["id"] for scenario in bench.scenarios}
    audio_variant_ids = set()
    if args.audio_manifest:
        with open(args.audio_manifest, encoding="utf-8") as f:
            audio_manifest = json.load(f)
        audio_variant_ids = {
            variant["id"]
            for variant in audio_manifest.get("variants", [])
            if isinstance(variant, dict) and variant.get("id")
        }
    issues = validate_split_manifest_file(
        args.splits,
        scenario_ids=scenario_ids,
        audio_variant_ids=audio_variant_ids,
    )
    if issues:
        print("Split manifest validation failed:")
        for issue in issues:
            print(f"  {issue.item_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)
    print(f"Validated split manifest: {args.splits}")


def cmd_split_commitments(args: argparse.Namespace) -> None:
    commitments = build_split_commitments_file(
        scenario_path=args.scenarios,
        split_path=args.splits,
        audio_manifest_path=args.audio_manifest,
        output_path=args.output,
        include_public_ids=not args.hide_public_ids,
        include_sealed_ids=args.reveal_sealed_ids,
    )
    _print_split_commitments(commitments)
    if args.output:
        print(f"\nSaved split commitments to {args.output}")


def cmd_validate_split_commitments(args: argparse.Namespace) -> None:
    issues = validate_split_commitments_file(
        args.commitments,
        scenario_path=args.scenarios,
        split_path=args.splits,
        audio_manifest_path=args.audio_manifest,
    )
    if issues:
        print("Split commitment validation failed:")
        for issue in issues:
            print(f"  {issue.item_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)
    print(f"Validated split commitments: {args.commitments}")


def cmd_datasheet(args: argparse.Namespace) -> None:
    datasheet = build_benchmark_datasheet_file(
        scenario_path=args.scenarios,
        audio_manifest_path=args.audio_manifest,
        pricing_manifest_path=args.pricing_manifest,
        split_manifest_path=args.splits,
        provenance_manifest_path=args.provenance,
        changelog_path=args.changelog,
        baseline_manifest_path=args.baseline_manifest,
        review_manifest_path=args.review_manifest,
        split_commitment_path=args.split_commitments,
        output_path=args.output,
    )
    _print_datasheet(datasheet)
    if args.output:
        print(f"\nSaved benchmark datasheet to {args.output}")


def cmd_validate_datasheet(args: argparse.Namespace) -> None:
    all_issues = []
    for datasheet_path in args.datasheets:
        issues = validate_benchmark_datasheet_file(datasheet_path)
        if issues:
            all_issues.append((datasheet_path, issues))

    if all_issues:
        print("Benchmark datasheet validation failed:")
        for datasheet_path, issues in all_issues:
            print(f"  {datasheet_path}:")
            for issue in issues:
                print(f"    {issue.path}: {issue.message}")
        raise SystemExit(1)

    print(f"Validated {len(args.datasheets)} benchmark datasheet file(s)")


def cmd_baselines(args: argparse.Namespace) -> None:
    manifest = build_reference_baselines(
        scenario_path=args.scenarios,
        audio_manifest_path=args.audio_manifest,
        output_dir=args.output_dir,
        manifest_path=args.output,
        trials=args.trials,
    )
    _print_baselines(manifest)
    print(f"\nSaved baseline manifest to {args.output}")


def cmd_validate_baselines(args: argparse.Namespace) -> None:
    all_issues = []
    for manifest_path in args.manifests:
        issues = validate_reference_baselines_file(manifest_path)
        if issues:
            all_issues.append((manifest_path, issues))

    if all_issues:
        print("Reference baseline validation failed:")
        for manifest_path, issues in all_issues:
            print(f"  {manifest_path}:")
            for issue in issues:
                print(f"    {issue.item_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)

    print(f"Validated {len(args.manifests)} reference baseline manifest file(s)")


def cmd_validate_audio_assets(args: argparse.Namespace) -> None:
    bench = OpenVoiceCSBench.load(args.scenarios)
    scenario_ids = {scenario["id"] for scenario in bench.scenarios}
    issues = validate_audio_assets_file(
        args.audio_manifest,
        root_dir=args.audio_root,
        scenario_ids=scenario_ids,
        require_sha256=not args.allow_missing_sha256,
        duration_tolerance_seconds=args.duration_tolerance_seconds,
    )
    if issues:
        print("Audio asset validation failed:")
        for issue in issues:
            print(f"  {issue.scenario_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)
    print(f"Validated audio assets: {args.audio_manifest}")


def cmd_validate_provenance(args: argparse.Namespace) -> None:
    bench = OpenVoiceCSBench.load(args.scenarios)
    scenario_ids = {scenario["id"] for scenario in bench.scenarios}
    audio_variant_ids = set()
    if args.audio_manifest:
        with open(args.audio_manifest, encoding="utf-8") as f:
            audio_manifest = json.load(f)
        audio_variant_ids = {
            variant["id"]
            for variant in audio_manifest.get("variants", [])
            if isinstance(variant, dict) and variant.get("id")
        }
    issues = validate_provenance_manifest_file(
        args.provenance,
        scenario_ids=scenario_ids,
        audio_variant_ids=audio_variant_ids,
    )
    if issues:
        print("Provenance manifest validation failed:")
        for issue in issues:
            print(f"  {issue.item_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)
    print(f"Validated provenance manifest: {args.provenance}")


def cmd_validate_changelog(args: argparse.Namespace) -> None:
    bench = OpenVoiceCSBench.load(args.scenarios)
    scenario_ids = {scenario["id"] for scenario in bench.scenarios}
    audio_variant_ids = set()
    if args.audio_manifest:
        with open(args.audio_manifest, encoding="utf-8") as f:
            audio_manifest = json.load(f)
        audio_variant_ids = {
            variant["id"]
            for variant in audio_manifest.get("variants", [])
            if isinstance(variant, dict) and variant.get("id")
        }
    issues = validate_changelog_file(
        args.changelog,
        scenario_ids=scenario_ids,
        audio_variant_ids=audio_variant_ids,
        benchmark_version=bench.version,
    )
    if issues:
        print("Changelog validation failed:")
        for issue in issues:
            print(f"  {issue.item_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)
    print(f"Validated changelog: {args.changelog}")


def cmd_validate_reviews(args: argparse.Namespace) -> None:
    bench = OpenVoiceCSBench.load(args.scenarios)
    scenario_ids = {scenario["id"] for scenario in bench.scenarios}
    issues = validate_review_manifest_file(
        args.review_manifest,
        scenario_ids=scenario_ids,
        benchmark_version=bench.version,
    )
    if issues:
        print("Scenario review manifest validation failed:")
        for issue in issues:
            print(f"  {issue.item_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)
    print(f"Validated scenario reviews: {args.review_manifest}")


def cmd_pin_audio_assets(args: argparse.Namespace) -> None:
    bench = OpenVoiceCSBench.load(args.scenarios)
    scenario_ids = {scenario["id"] for scenario in bench.scenarios}
    output_path = args.audio_manifest if args.in_place else args.output
    if output_path is None:
        print("pin-audio-assets requires --output or --in-place")
        raise SystemExit(2)

    result = pin_audio_manifest_assets_file(
        args.audio_manifest,
        output_path=output_path,
        root_dir=args.audio_root,
        scenario_ids=scenario_ids,
    )
    if result["issues"]:
        print("Audio asset pinning failed:")
        for issue in result["issues"]:
            print(f"  {issue.scenario_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)

    summary = result["summary"]
    print(
        "Pinned audio assets:   "
        f"{summary['num_pinned']}/{summary['num_variants']} variants"
    )
    print(f"Total duration:        {summary['total_duration_seconds']} seconds")
    print(f"Saved manifest to:     {result['output_path']}")


def cmd_add_scenarios(args: argparse.Namespace) -> None:
    result = add_scenarios_to_release_files(
        draft_path=args.draft,
        scenario_path=args.scenarios,
        split_path=args.splits,
        provenance_path=args.provenance,
        output_scenario_path=args.output_scenarios,
        output_split_path=args.output_splits,
        output_provenance_path=args.output_provenance,
        audio_manifest_path=args.audio_manifest,
        split=args.split,
        source_type=args.source_type,
        license_id=args.license,
        authoring_method=args.authoring_method,
        contamination_risk=args.contamination_risk,
        review_status=args.review_status,
    )
    if result["issues"]:
        print("Scenario expansion failed:")
        for issue in result["issues"]:
            print(f"  {issue.item_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)
    print(f"Added scenarios:       {result['num_added']}")
    for scenario_id in result["added_ids"]:
        print(f"  {scenario_id}")
    print("Saved files:")
    for label, path in result["outputs"].items():
        print(f"  {label}: {path}")


def cmd_scaffold_scenarios(args: argparse.Namespace) -> None:
    drafts = scaffold_scenario_drafts(
        scenario_path=args.scenarios,
        split_path=args.splits,
        target_path=args.targets,
        profile=args.profile,
        count=args.count,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(drafts, f, indent=2)
        f.write("\n")
    print(f"Scaffolded scenarios:  {drafts['num_scenarios']}")
    print(f"Profile:               {drafts['profile']}")
    print(f"Saved draft file to:   {output}")


def cmd_coverage_plan(args: argparse.Namespace) -> None:
    plan = build_coverage_plan(
        scenario_path=args.scenarios,
        split_path=args.splits,
        audio_manifest_path=args.audio_manifest,
        target_path=args.targets,
        profile=args.profile,
    )
    _print_coverage_plan(plan)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2)
        print(f"\nSaved coverage plan to {output}")
    if args.strict and not plan["passed"]:
        raise SystemExit(1)


def cmd_readiness(args: argparse.Namespace) -> None:
    audit = build_release_audit(
        scenario_path=args.scenarios,
        audio_manifest_path=args.audio_manifest,
        audio_asset_root=args.audio_root,
        pricing_manifest_path=args.pricing_manifest,
        split_manifest_path=args.splits,
        split_commitment_path=args.split_commitments,
        provenance_manifest_path=args.provenance,
        changelog_path=args.changelog,
        baseline_manifest_path=args.baseline_manifest,
        review_manifest_path=args.review_manifest,
        judge_protocol_path=args.judge_protocol,
        judge_study_path=args.judge_study,
        judge_annotation_package_path=args.judge_annotation_package,
        sealed_ops_path=args.sealed_ops,
        sealed_queue_path=args.sealed_queue,
        external_endpoint_contract_path=args.external_endpoint_contract,
        external_systems_path=args.external_systems,
        claims_manifest_path=args.claims,
        submission_intake_path=args.submission_intake,
    )
    frontier_report = _load_json(args.frontier_report) if args.frontier_report else None
    run_manifest_path = Path(args.run_manifest) if args.run_manifest else None
    run_manifest = _load_json(run_manifest_path) if run_manifest_path else None
    readiness = evaluate_release_readiness(
        audit,
        profile=args.profile,
        frontier_report=frontier_report,
        run_manifest=run_manifest,
        run_manifest_base_dir=run_manifest_path.parent if run_manifest_path else ".",
        verify_run_manifest_files=run_manifest_path is not None,
        plot_dir=args.plot_dir,
    )
    _print_readiness(readiness)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(readiness, f, indent=2)
        print(f"\nSaved readiness report to {output}")
    if args.strict and not readiness["passed"]:
        raise SystemExit(1)


def cmd_judge_report(args: argparse.Namespace) -> None:
    report = build_judge_report_from_files(
        args.report,
        args.annotations,
        rubric_path=args.rubric,
    )
    _print_judge_report(report)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved judge report to {output}")


def cmd_apply_judge_report(args: argparse.Namespace) -> None:
    report = apply_judge_report_from_files(args.report, args.judge_report)
    issues = validate_report(report)
    if issues:
        print("Judged report validation failed:")
        for issue in issues:
            print(f"  {issue.scenario_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)

    _print_judged_benchmark_report(report)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved judged benchmark report to {output}")


def cmd_model_judge(args: argparse.Namespace) -> None:
    load_workspace_env(args.env)
    with open(args.report, encoding="utf-8") as f:
        source_report = json.load(f)
    with open(args.rubric, encoding="utf-8") as f:
        rubric = json.load(f)
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    judge_specs = [parse_model_judge_spec(value) for value in args.judge]
    if len(judge_specs) < 2:
        print("model-judge requires at least two --judge specs for audited model judging")
        raise SystemExit(2)
    adjudicator = parse_model_judge_spec(args.adjudicator) if args.adjudicator else None
    disagreement_threshold = args.disagreement_threshold
    if disagreement_threshold is None and args.judge_protocol:
        protocol = load_judge_protocol(args.judge_protocol)
        threshold = protocol.get("adjudication", {}).get("disagreement_threshold")
        if isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
            disagreement_threshold = float(threshold)

    annotations = generate_model_judge_annotations(
        source_report,
        judge_specs=judge_specs,
        rubric=rubric,
        prompt=prompt,
        adjudicator=adjudicator,
        disagreement_threshold=disagreement_threshold,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        timeout_seconds=args.timeout_seconds,
    )
    annotations_output = Path(args.annotations_output)
    write_judge_annotations_jsonl(annotations, annotations_output)

    judge_report = build_judge_report(source_report, annotations, rubric)
    judge_report_output = Path(args.judge_report_output)
    judge_report_output.parent.mkdir(parents=True, exist_ok=True)
    with open(judge_report_output, "w", encoding="utf-8") as f:
        json.dump(judge_report, f, indent=2)

    judged_report = None
    judged_report_output = Path(args.judged_report_output) if args.judged_report_output else None
    if judged_report_output:
        judged_report = apply_judge_report(source_report, judge_report)
        issues = validate_report(judged_report)
        if issues:
            print("Judged report validation failed:")
            for issue in issues:
                print(f"  {issue.scenario_id}::{issue.path}: {issue.message}")
            raise SystemExit(1)
        judged_report_output.parent.mkdir(parents=True, exist_ok=True)
        with open(judged_report_output, "w", encoding="utf-8") as f:
            json.dump(judged_report, f, indent=2)

    _print_model_judge_result(
        annotations=annotations,
        annotations_output=annotations_output,
        judge_report=judge_report,
        judge_report_output=judge_report_output,
        judged_report=judged_report,
        judged_report_output=judged_report_output,
    )


def cmd_compare(args: argparse.Namespace) -> None:
    validation_inputs = [args.baseline, args.candidate]
    all_issues = []
    for report_path in validation_inputs:
        issues = validate_report_file(report_path)
        if issues:
            all_issues.append((report_path, issues))
    if all_issues:
        print("Report validation failed:")
        for report_path, issues in all_issues:
            print(f"  {report_path}:")
            for issue in issues:
                print(f"    {issue.scenario_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)

    baseline = _load_json(args.baseline)
    candidate = _load_json(args.candidate)
    comparison = compare_reports(
        baseline,
        candidate,
        iterations=args.iterations,
        seed=args.seed,
        confidence=args.confidence,
    )
    _print_comparison(comparison)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nSaved comparison report to {output}")


def cmd_score(args: argparse.Namespace) -> None:
    bench = OpenVoiceCSBench.load(args.scenarios)
    agent_fn = _resolve_agent(args.agent)
    report = bench.score_agent(
        agent_fn,
        max_scenarios=args.max,
        trials=args.trials,
        track=args.track,
        model_metadata=_reference_model_metadata(args.agent),
    )
    _print_report(report)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report to {output}")


def cmd_score_audio(args: argparse.Namespace) -> None:
    bench = OpenVoiceCSBench.load(args.scenarios)
    agent_fn = _resolve_agent(args.agent)
    report = bench.score_audio_manifest(
        agent_fn,
        manifest_path=args.audio_manifest,
        max_variants=args.max,
        trials=args.trials,
        track=args.track,
        model_metadata=_reference_model_metadata(
            args.agent,
            input_modality="audio",
        ),
    )
    _print_report(report)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report to {output}")


def cmd_leaderboard(args: argparse.Namespace) -> None:
    reports = _load_validated_reports(args.reports)
    if not reports:
        print("No reports matched.")
        raise SystemExit(1)
    leaderboard = build_leaderboard(reports)
    _print_leaderboard(leaderboard)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(leaderboard, f, indent=2)
        print(f"\nSaved leaderboard to {output}")


def cmd_frontier(args: argparse.Namespace) -> None:
    reports = _load_validated_reports(args.reports)
    if not reports:
        print("No reports matched.")
        raise SystemExit(1)
    pricing_manifest = (
        load_pricing_manifest(args.pricing_manifest)
        if args.pricing_manifest
        else None
    )
    frontier = build_frontier_report(
        reports,
        pricing_manifest=pricing_manifest,
        pricing_snapshot_date=args.pricing_snapshot_date,
        environment={
            "region": args.region,
            "network": args.network,
            "hardware_profile": args.hardware_profile,
            "transport": args.transport,
            "concurrency_levels": args.concurrency_levels,
        },
        experience_gate=args.experience_gate,
        utility_weights=_parse_utility_weights(args.utility) if args.utility else None,
        latency_targets_ms=args.latency_target_ms,
        cost_targets_usd=args.cost_target_usd,
    )
    _print_frontier(frontier)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(frontier, f, indent=2)
        print(f"\nSaved frontier report to {output}")
    if args.plot_dir:
        artifacts = write_frontier_artifacts(frontier, args.plot_dir)
        print(f"\nSaved frontier plot artifacts to {args.plot_dir}")
        for kind, path in artifacts.items():
            print(f"  {kind}: {path}")
    if args.scorecard_dir:
        artifacts = write_scorecard_artifacts(frontier, args.scorecard_dir)
        print(f"\nSaved scorecard artifacts to {args.scorecard_dir}")
        for kind, path in artifacts.items():
            print(f"  {kind}: {path}")


def cmd_load(args: argparse.Namespace) -> None:
    bench = OpenVoiceCSBench.load(args.scenarios)
    transport = args.transport or ("websocket" if args.endpoint else "in_process")
    if transport == "websocket":
        if not args.endpoint:
            raise SystemExit("--endpoint is required for websocket transport")
        client = WebSocketRealtimeClient(args.endpoint, timeout_seconds=args.timeout_seconds)
        metadata = _endpoint_model_metadata(args.endpoint, transport="websocket")
    elif transport == "webrtc":
        if not args.endpoint:
            raise SystemExit("--endpoint is required for webrtc transport")
        client = WebRTCRealtimeClient(args.endpoint, timeout_seconds=args.timeout_seconds)
        metadata = _endpoint_model_metadata(args.endpoint, transport="webrtc")
    elif transport == "in_process":
        client = ReferenceRealtimeClient(builtin_realtime_agent(args.agent))
        metadata = _reference_model_metadata(args.agent, transport="in_process")
    else:
        raise SystemExit(f"unsupported transport: {transport}")

    report = run_openvoicecs_realtime_load(
        bench,
        client,
        max_scenarios=args.max,
        trials=args.trials,
        track=args.track,
        concurrency_levels=tuple(args.concurrency_levels),
        region=args.region,
        network=args.network,
        hardware_profile=args.hardware_profile,
        seed=args.seed,
        pricing_snapshot_date=args.pricing_snapshot_date,
        model_metadata=metadata,
    )
    _print_load_report(report)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved realtime load report to {output}")


def cmd_audit(args: argparse.Namespace) -> None:
    audit = build_release_audit(
        scenario_path=args.scenarios,
        audio_manifest_path=args.audio_manifest,
        audio_asset_root=args.audio_root,
        pricing_manifest_path=args.pricing_manifest,
        split_manifest_path=args.splits,
        split_commitment_path=args.split_commitments,
        provenance_manifest_path=args.provenance,
        changelog_path=args.changelog,
        baseline_manifest_path=args.baseline_manifest,
        review_manifest_path=args.review_manifest,
        judge_protocol_path=args.judge_protocol,
        judge_study_path=args.judge_study,
        judge_annotation_package_path=args.judge_annotation_package,
        sealed_ops_path=args.sealed_ops,
        sealed_queue_path=args.sealed_queue,
        external_endpoint_contract_path=args.external_endpoint_contract,
        external_systems_path=args.external_systems,
        claims_manifest_path=args.claims,
        submission_intake_path=args.submission_intake,
    )
    _print_audit(audit)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2)
        print(f"\nSaved audit report to {output}")


def cmd_run_manifest(args: argparse.Namespace) -> None:
    manifest = build_run_manifest(
        args.reports,
        scenario_path=args.scenarios,
        audio_manifest_path=args.audio_manifest,
        audio_asset_root=args.audio_root,
        pricing_manifest_path=args.pricing_manifest,
        split_manifest_path=args.splits,
        provenance_manifest_path=args.provenance,
        changelog_path=args.changelog,
        baseline_manifest_path=args.baseline_manifest,
        review_manifest_path=args.review_manifest,
        judge_model=args.judge_model,
        judge_prompt_path=args.judge_prompt,
        seed=args.seed,
        region=args.region,
        network=args.network,
        hardware_profile=args.hardware_profile,
        transport=args.transport,
        concurrency_levels=args.concurrency_levels,
    )
    _print_run_manifest(manifest)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nSaved run manifest to {output}")


def cmd_release_bundle(args: argparse.Namespace) -> None:
    bundle = build_frontier_release_bundle(
        args.reports,
        args.output_dir,
        scenario_path=args.scenarios,
        audio_manifest_path=args.audio_manifest,
        audio_asset_root=args.audio_root,
        pricing_manifest_path=args.pricing_manifest,
        split_manifest_path=args.splits,
        provenance_manifest_path=args.provenance,
        changelog_path=args.changelog,
        baseline_manifest_path=args.baseline_manifest,
        review_manifest_path=args.review_manifest,
        judge_model=args.judge_model,
        judge_prompt_path=args.judge_prompt,
        seed=args.seed,
        region=args.region,
        network=args.network,
        hardware_profile=args.hardware_profile,
        transport=args.transport,
        concurrency_levels=args.concurrency_levels,
        pricing_snapshot_date=args.pricing_snapshot_date,
        experience_gate=args.experience_gate,
        utility_weights=_parse_utility_weights(args.utility) if args.utility else None,
        latency_targets_ms=args.latency_target_ms,
        cost_targets_usd=args.cost_target_usd,
        readiness_profile=args.readiness_profile,
    )
    _print_release_bundle(bundle)
    if args.strict and not bundle["validation"]["passed"]:
        raise SystemExit(1)


def cmd_submit(args: argparse.Namespace) -> None:
    report = score_submission(
        args.submission,
        scenario_path=args.scenarios,
        audio_manifest_path=args.audio_manifest,
        mode=args.mode,
        max_items=args.max,
        trials=args.trials,
        track=args.track,
        submission_name=args.name,
        provider=args.provider,
        model_id=args.model_id,
        pricing_profile_id=args.pricing_profile_id,
        pricing_snapshot_date=args.pricing_snapshot_date,
        pipeline_type=args.pipeline_type,
    )
    _print_report(report)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved submission report to {output}")


def cmd_score_provider(args: argparse.Namespace) -> None:
    pricing = {}
    if args.input_price is not None:
        pricing["input_per_mtok"] = args.input_price
    if args.output_price is not None:
        pricing["output_per_mtok"] = args.output_price
    spec = build_provider_spec(
        args.provider,
        model_id=args.model,
        display_name=args.name,
        api_key=args.api_key,
        base_url=args.base_url,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        native_tools=args.native_tools,
        pricing=pricing or None,
    )
    report = score_provider(
        spec,
        scenario_path=args.scenarios,
        audio_manifest_path=args.audio_manifest,
        mode=args.mode,
        max_items=args.max,
        trials=args.trials,
        track=args.track,
        pricing_profile_id=args.pricing_profile_id,
        pricing_snapshot_date=args.pricing_snapshot_date,
    )
    _print_report(report)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved provider report to {output}")


def cmd_submit_endpoint(args: argparse.Namespace) -> None:
    try:
        headers = _parse_headers(args.header or [])
    except ValueError as exc:
        print(f"Header parsing failed: {exc}")
        raise SystemExit(2) from exc
    report = score_external_endpoint(
        args.endpoint,
        scenario_path=args.scenarios,
        audio_manifest_path=args.audio_manifest,
        mode=args.mode,
        max_items=args.max,
        trials=args.trials,
        track=args.track,
        headers=headers,
        timeout_seconds=args.timeout_seconds,
        run_id=args.run_id,
        model_metadata={
            "display_name": args.name or args.endpoint,
            "provider": args.provider,
            "model_id": args.model_id,
            "pricing_profile_id": args.pricing_profile_id,
            "pricing_snapshot_date": args.pricing_snapshot_date,
            "pipeline_type": args.pipeline_type,
        },
    )
    _print_report(report)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved endpoint submission report to {output}")


def cmd_submission_card(args: argparse.Namespace) -> None:
    card = build_submission_card_from_file(
        args.report,
        submitter_name=args.submitter,
        organization=args.organization,
        contact=args.contact,
        repository_url=args.repository_url,
        license_id=args.license,
        training_data_statement=args.training_data_statement,
        safety_statement=args.safety_statement,
        limitations=args.limitations or [],
    )
    _print_submission_card(card)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2)
        print(f"\nSaved submission card to {output}")


def cmd_validate_submission_card(args: argparse.Namespace) -> None:
    all_issues = []
    for card_path in args.cards:
        issues = validate_submission_card_file(card_path)
        if issues:
            all_issues.append((card_path, issues))
    if all_issues:
        print("Submission card validation failed:")
        for card_path, issues in all_issues:
            print(f"  {card_path}:")
            for issue in issues:
                print(f"    {issue.item_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)
    print(f"Validated {len(args.cards)} submission card file(s)")


def cmd_validate_submission_intake(args: argparse.Namespace) -> None:
    issues = validate_submission_intake_file(args.intake)
    if issues:
        print("Submission intake validation failed:")
        for issue in issues:
            print(f"  {issue.item_id}::{issue.path}: {issue.message}")
        raise SystemExit(1)
    manifest = _load_json(args.intake)
    stats = submission_intake_stats(manifest)
    print(
        "Validated submission intake: "
        f"{stats.get('submission_id')} "
        f"status={stats.get('status')} "
        f"artifacts={stats.get('required_artifacts_present')}/{len(manifest.get('artifacts', {}))}"
    )


def cmd_init_submission(args: argparse.Namespace) -> None:
    try:
        output = write_submission_template(
            args.output,
            function_name=args.function_name,
            overwrite=args.overwrite,
        )
    except (FileExistsError, ValueError) as exc:
        print(f"Submission template creation failed: {exc}")
        raise SystemExit(1) from exc
    print(f"Wrote submission adapter template to {output}")
    print(
        "Score it with: "
        f"python scripts/run_openvoicecs.py submit {output}:{args.function_name} "
        "--name my_submission --trials 1"
    )


def _resolve_agent(name: str):
    if name == "oracle":
        return oracle_agent
    if name == "noop":
        return no_op_agent
    raise ValueError(f"Unknown built-in agent: {name}")


def _reference_model_metadata(
    agent: str,
    *,
    input_modality: str = "text",
    transport: str | None = None,
) -> dict[str, Any]:
    metadata = {
        "display_name": agent,
        "agent": agent,
        "provider": "reference",
        "model_id": f"{agent}-agent-v0.1",
        "pricing_profile_id": "reference-zero-v0.1",
        "pricing_snapshot_date": "2026-06-11",
        "pipeline_type": "cascaded",
        "input_modality": input_modality,
    }
    if transport:
        metadata["transport"] = transport
    return metadata


def _endpoint_model_metadata(endpoint: str, *, transport: str) -> dict[str, Any]:
    return {
        "display_name": endpoint,
        "endpoint": endpoint,
        "provider": "external_endpoint",
        "model_id": endpoint,
        "pricing_profile_id": "reference-zero-v0.1",
        "pricing_snapshot_date": "2026-06-11",
        "pipeline_type": "cascaded",
        "transport": transport,
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return data


def _load_validated_reports(patterns: list[str]) -> list[dict[str, Any]]:
    try:
        return load_reports(patterns, validate=True)
    except ValueError as exc:
        print(exc)
        raise SystemExit(1) from exc


def _print_report(report: dict[str, Any]) -> None:
    print("\nOPENVOICECS-BENCH RESULTS")
    print("=" * 72)
    print(f"Score:                 {report['overall_score']:.2f} / 100")
    print(f"Scenarios:             {report['num_scenarios']}")
    print(f"Trials per scenario:   {report['num_trials_per_scenario']}")
    print(f"pass@k:                {report['pass_at_k']:.1%}")
    print(f"pass^k:                {report['pass_k']:.1%}")
    print(f"Mean pass rate:        {report['mean_pass_rate']:.1%}")
    gates = report.get("reliability_gates", {})
    pass_k_gate = gates.get("pass_k", {})
    if pass_k_gate:
        status = "pass" if pass_k_gate.get("passed") else "fail"
        print(
            f"pass^k gate:           {status} "
            f"(min {pass_k_gate.get('minimum_for_leaderboard', 0):.0%})"
        )
    ci = report.get("confidence_intervals", {}).get("trial_pass_rate", {})
    if ci.get("low") is not None:
        print(f"Trial pass 95% CI:     {ci['low']:.1%} - {ci['high']:.1%}")

    ops = report.get("operational_metrics", {})
    if ops:
        print(f"Median latency:        {_fmt(ops.get('median_latency_ms'), ' ms')}")
        print(f"P95 latency:           {_fmt(ops.get('p95_latency_ms'), ' ms')}")
        print(f"Avg tool calls:        {_fmt(ops.get('avg_tool_calls'))}")
        print(f"Avg wasted tools:      {_fmt(ops.get('avg_wasted_tool_calls'))}")
    stability = report.get("stability_metrics", {})
    if stability:
        print(f"Scenario flake rate:   {_fmt_pct(stability.get('scenario_flake_rate'))}")
        print(f"Unstable scenarios:    {stability.get('unstable_scenario_count', 0)}")
        print(f"Tool failure recovery: {_fmt_pct(stability.get('tool_failure_recovery_rate'))}")

    print("\nMetric scores:")
    for metric, score in report["metric_scores"].items():
        print(f"  {metric:18s} {score:.1%}")

    print("\nDomain breakdown:")
    for domain, item in report.get("domain_breakdown", {}).items():
        print(
            f"  {domain:18s} n={item['count']:<2d} "
            f"pass@k={item['pass_at_k']:.0%} pass^k={item['pass_k']:.0%}"
        )

    failed = [result for result in report["results"] if not result["pass_k"]]
    if failed:
        print("\nFailures:")
        for result in failed:
            first_failed = next(
                (trial for trial in result["trials"] if not trial.get("passed", False)),
                None,
            )
            reason = "unknown"
            if first_failed:
                missing_state = first_failed.get("state_check", {}).get("missing_or_wrong", [])
                missing_tools = first_failed.get("tool_check", {}).get("missing_expected", [])
                missing_events = first_failed.get("policy_check", {}).get("missing_required", [])
                if missing_state:
                    reason = f"state: {missing_state[0]['path']}"
                elif missing_tools:
                    reason = f"tool: {missing_tools[0]['name']}"
                elif missing_events:
                    reason = f"event: {missing_events[0]}"
                elif first_failed.get("safety_check", {}).get("violations"):
                    reason = f"safety: {first_failed['safety_check']['violations'][0]['type']}"
            print(f"  {result['id']}: {reason}")
    failure_analysis = report.get("failure_analysis", {})
    categories = failure_analysis.get("categories", {})
    if categories:
        print("\nFailure categories:")
        for category, count in categories.items():
            print(f"  {category:24s} {count}")


def _print_leaderboard(leaderboard: dict[str, Any]) -> None:
    print("\nOPENVOICECS-BENCH LEADERBOARD")
    print("=" * 88)
    print(f"{'#':>2}  {'Submission':<28} {'Score':>7} {'pass^k':>7} {'pass@k':>7} {'P50 ms':>9}")
    print("-" * 88)
    for rank, name in enumerate(leaderboard["ranking"], 1):
        model = leaderboard["models"][name]
        ops = model.get("operational_metrics", {})
        print(
            f"{rank:>2}  {name:<28.28s} "
            f"{model.get('overall_score', 0):>7.2f} "
            f"{model.get('pass_k', 0):>7.1%} "
            f"{model.get('pass_at_k', 0):>7.1%} "
            f"{_fmt(ops.get('median_latency_ms')):>9}"
        )


def _print_comparison(comparison: dict[str, Any]) -> None:
    print("\nOPENVOICECS-BENCH PAIRED COMPARISON")
    print("=" * 88)
    print(f"Baseline:              {comparison['baseline']}")
    print(f"Candidate:             {comparison['candidate']}")
    matched = comparison["matched_scenarios"]
    print(f"Matched scenarios:     {matched['count']}")
    if matched.get("baseline_only") or matched.get("candidate_only"):
        print(
            "Unmatched scenarios:   "
            f"baseline_only={len(matched.get('baseline_only', []))}, "
            f"candidate_only={len(matched.get('candidate_only', []))}"
        )

    summary = comparison["summary"]
    score_delta = summary["mean_paired_scenario_score_delta"]
    pass_delta = summary["mean_paired_pass_rate_delta"]
    mcnemar = summary["mcnemar_exact"]
    print(f"Overall score delta:   {summary['overall_score_delta']:+.3f}")
    print(
        "Scenario score delta:  "
        f"{score_delta['mean']:+.3f} "
        f"[{score_delta['ci_low']:+.3f}, {score_delta['ci_high']:+.3f}]"
    )
    print(
        "Pass-rate delta:       "
        f"{pass_delta['mean']:+.3f} "
        f"[{pass_delta['ci_low']:+.3f}, {pass_delta['ci_high']:+.3f}]"
    )
    stratified = comparison.get("stratified_deltas", {})
    if stratified:
        print("\nStratified scenario-score deltas:")
        for field in ("domain", "track", "difficulty"):
            score_delta = stratified.get(field, {}).get("scenario_score_delta")
            if not score_delta:
                continue
            print(
                f"  by {field:<10} "
                f"{score_delta['mean']:+.3f} "
                f"[{score_delta['ci_low']:+.3f}, {score_delta['ci_high']:+.3f}] "
                f"strata={score_delta['num_strata']}"
            )
    print(
        "McNemar exact:         "
        f"candidate_wins={mcnemar['candidate_wins']} "
        f"baseline_wins={mcnemar['baseline_wins']} "
        f"p={mcnemar['p_value']}"
    )
    print(f"Interpretation:        {summary['interpretation']}")

    print("\nMetric deltas:")
    for metric, item in comparison["metric_deltas"].items():
        print(
            f"  {metric:18s} "
            f"{item['mean']:+.4f} [{item['ci_low']:+.4f}, {item['ci_high']:+.4f}]"
        )


def _print_frontier(frontier: dict[str, Any]) -> None:
    print("\nLATENCY-COST-QUALITY FRONTIER")
    print("=" * 104)
    print(f"Frontier systems: {', '.join(frontier['frontier']) or 'none'}")
    print(
        f"{'System':<28} {'Eligible':>8} {'P95 TTFB':>10} "
        f"{'$/success':>12} {'Task':>8} {'Exp':>8} {'@100 p95':>10}"
    )
    print("-" * 104)
    for name, scorecard in frontier["scorecards"].items():
        system = frontier["systems"][name]
        print(
            f"{name:<28.28s} "
            f"{str(system['frontier_eligible']):>8} "
            f"{_fmt(scorecard.get('p95_v2v_ttfb_ms')):>10} "
            f"{_fmt(scorecard.get('cost_usd_per_successful_conversation')):>12} "
            f"{_fmt_pct(scorecard.get('task_success_rate')):>8} "
            f"{_fmt_pct(scorecard.get('experience_score')):>8} "
            f"{_fmt(scorecard.get('latency_at_100_concurrency_p95_ms')):>10}"
        )
        if system["exclusion_reasons"]:
            reasons = ", ".join(reason["type"] for reason in system["exclusion_reasons"])
            print(f"{'':<28} excluded: {reasons}")
    constrained = frontier.get("constrained_frontiers", {}).get("entries", [])
    if constrained:
        print("\nConstrained frontiers:")
        for entry in constrained:
            constraint = entry["constraint"]
            latency = constraint.get("latency_ms")
            cost = constraint.get("cost_usd")
            label = []
            if latency is not None:
                label.append(f"p95<={latency:g}ms")
            if cost is not None:
                label.append(f"cost<={cost:g}")
            print(f"  {' / '.join(label)}: {', '.join(entry['frontier']) or 'none'}")


def _print_load_report(report: dict[str, Any]) -> None:
    print("\nOPENVOICECS REALTIME LOAD RESULTS")
    print("=" * 88)
    print(f"Scenarios:             {report['num_scenarios']}")
    print(f"Trials per scenario:   {report['num_trials_per_scenario']}")
    print(f"pass^k:                {report['pass_k']:.1%}")
    print(f"Mean pass rate:        {report['mean_pass_rate']:.1%}")
    env = report.get("reference_client", {})
    print(f"Transport:             {env.get('transport', 'n/a')}")
    print(f"Region/network:        {env.get('region', 'n/a')} / {env.get('network', 'n/a')}")
    print(f"Hardware profile:      {env.get('hardware_profile', 'n/a')}")
    ttfb = report.get("operational_metrics", {}).get("v2v_ttfb_ms", {})
    print(
        "V2V TTFB p50/p95/p99: "
        f"{_fmt(ttfb.get('p50'), ' ms')} / "
        f"{_fmt(ttfb.get('p95'), ' ms')} / "
        f"{_fmt(ttfb.get('p99'), ' ms')}"
    )
    print("Load:")
    for level, item in report.get("operational_metrics", {}).get("load", {}).items():
        print(
            f"  concurrency={level:<4} "
            f"count={item.get('count', 0):<4} "
            f"p95={_fmt(item.get('p95'), ' ms')} "
            f"requested={item.get('requested_calls', 'n/a')} "
            f"completed={item.get('completed_calls', 'n/a')} "
            f"errors={item.get('error_calls', 'n/a')} "
            f"peak={item.get('peak_active_calls', 'n/a')} "
            f"saturated={item.get('saturated', 'n/a')}"
        )


def _print_audit(audit: dict[str, Any]) -> None:
    print("\nOPENVOICECS-BENCH RELEASE AUDIT")
    print("=" * 88)
    print(f"Version:               {audit['version']}")
    print(f"Validation passed:     {audit['validation']['passed']}")
    print(f"Release gates passed:  {audit['release_gates']['passed']}")
    stats = audit["scenario_stats"]
    print(f"Scenarios:             {stats['num_scenarios']}")
    print(f"Audio variants:        {audit['audio_manifest_stats']['num_variants']}")
    assets = audit.get("audio_asset_stats", {})
    print(
        "Audio files verified:  "
        f"{assets.get('num_sha256_verified', 0)}/{assets.get('num_variants', 0)}"
    )
    pricing = audit.get("pricing_manifest_stats", {})
    print(f"Pricing snapshot:      {pricing.get('snapshot_date', 'n/a')}")
    print(f"Pricing profiles:      {pricing.get('num_profiles', 0)}")
    splits = audit.get("split_manifest_stats", {})
    print(f"Split manifest:        {splits.get('version', 'n/a')}")
    print(f"Scenario split cover:  {_fmt_pct(splits.get('scenario_coverage'))}")
    print(f"Audio split cover:     {_fmt_pct(splits.get('audio_variant_coverage'))}")
    provenance = audit.get("provenance_stats", {})
    print(f"Provenance manifest:   {provenance.get('version', 'n/a')}")
    print(f"Scenario provenance:   {_fmt_pct(provenance.get('scenario_coverage'))}")
    print(f"Audio provenance:      {_fmt_pct(provenance.get('audio_variant_coverage'))}")
    changelog = audit.get("changelog_stats", {})
    print(f"Changelog:             {changelog.get('version', 'n/a')}")
    print(f"Open errata:           {changelog.get('num_open_errata', 'n/a')}")
    baselines = audit.get("baseline_stats", {})
    print(f"Reference baselines:   {baselines.get('num_baselines', 'n/a')}")
    reviews = audit.get("review_stats", {})
    print(f"Scenario review cover: {_fmt_pct(reviews.get('scenario_approval_coverage'))}")
    print("Tracks:")
    for track, count in stats["tracks"].items():
        print(f"  {track}: {count}")
    print("Oracle coverage:")
    for name, item in audit["oracle_coverage"].items():
        print(f"  {name:22s} {item['count']}/{item['total']} ({item['rate']:.0%})")
    if audit["validation"]["issues"]:
        print("\nValidation issues:")
        for issue in audit["validation"]["issues"]:
            print(f"  {issue['scenario_id']}::{issue['path']}: {issue['message']}")


def _print_readiness(readiness: dict[str, Any]) -> None:
    print("\nOPENVOICECS-BENCH RELEASE READINESS")
    print("=" * 88)
    print(f"Profile:               {readiness['profile']}")
    print(f"Version:               {readiness.get('version', 'n/a')}")
    print(f"Passed:                {readiness['passed']}")
    print(f"Issues:                {readiness['num_issues']}")
    if readiness["issues"]:
        print("\nBlocking criteria:")
        for issue in readiness["issues"]:
            print(
                f"  {issue['criterion']}: observed={issue['observed']} "
                f"required={issue['required']} - {issue['message']}"
            )


def _print_split_commitments(commitments: dict[str, Any]) -> None:
    print("\nOPENVOICECS-BENCH SPLIT COMMITMENTS")
    print("=" * 88)
    print(f"Version:               {commitments.get('version')}")
    print(f"Hash algorithm:        {commitments.get('hash_algorithm')}")
    print(f"Root hash:             {commitments.get('root_hash')}")
    privacy = commitments.get("privacy", {})
    print(f"Public IDs revealed:   {privacy.get('public_dev_ids_revealed')}")
    print(f"Sealed IDs revealed:   {privacy.get('sealed_test_ids_revealed')}")
    print("Splits:")
    for split_name, split in commitments.get("splits", {}).items():
        print(
            f"  {split_name}: "
            f"scenarios={split['num_scenarios']} "
            f"audio_variants={split['num_audio_variants']}"
        )


def _print_datasheet(datasheet: dict[str, Any]) -> None:
    print("\nOPENVOICECS-BENCH DATASHEET")
    print("=" * 88)
    release = datasheet["release"]
    summary = datasheet["data_summary"]
    split_policy = datasheet["split_policy"]
    changelog = datasheet.get("changelog", {})
    baselines = datasheet.get("baselines", {})
    reviews = datasheet.get("scenario_reviews", {})
    validation = datasheet["release_validation"]
    print(f"Version:               {release.get('benchmark_version')}")
    print(f"Release stage:         {release.get('release_stage')}")
    print(f"Scenarios:             {summary.get('num_scenarios')}")
    print(f"Audio variants:        {summary.get('num_audio_variants')}")
    print(f"Split manifest:        {split_policy.get('manifest_version')}")
    print(f"Commitment root:       {split_policy.get('commitment_root_hash') or 'n/a'}")
    print(f"Changelog entries:     {changelog.get('num_entries', 'n/a')}")
    print(f"Open errata:           {changelog.get('num_open_errata', 'n/a')}")
    print(f"Reference baselines:   {baselines.get('num_baselines', 'n/a')}")
    print(f"Scenario reviews:      {_fmt_pct(reviews.get('scenario_approval_coverage'))}")
    print(f"Audit passed:          {validation.get('audit_passed')}")
    print(f"Release gates passed:  {validation.get('release_gates_passed')}")


def _print_baselines(manifest: dict[str, Any]) -> None:
    print("\nOPENVOICECS-BENCH REFERENCE BASELINES")
    print("=" * 88)
    print(f"Version:               {manifest.get('version')}")
    print(f"Benchmark version:     {manifest.get('benchmark_version')}")
    print(f"Baselines:             {len(manifest.get('baselines', []))}")
    for baseline in manifest.get("baselines", []):
        expected = baseline.get("expected", {})
        report = baseline.get("report", {})
        print(
            f"  {baseline.get('id'):24s} "
            f"score={_fmt(expected.get('overall_score')):>6} "
            f"pass^k={_fmt_pct(expected.get('pass_k')):>7} "
            f"report={report.get('sha256', 'n/a')[:12]}"
        )


def _print_coverage_plan(plan: dict[str, Any]) -> None:
    print("\nOPENVOICECS-BENCH COVERAGE PLAN")
    print("=" * 88)
    print(f"Profile:               {plan['profile']}")
    print(f"Target version:        {plan.get('target_version', 'n/a')}")
    print(f"Passed:                {plan['passed']}")
    total = plan["gaps"]["total"]
    print(
        "Scenarios:             "
        f"{total['current']}/{total['target']} "
        f"(need {total['needed']})"
    )
    for group_name in ("domains", "tracks", "difficulty", "splits"):
        print(f"{group_name.title()}:")
        for key, item in plan["gaps"][group_name].items():
            print(
                f"  {key:28s} {item['current']:>4}/{item['target']:<4} "
                f"need {item['needed']}"
            )
    audio = plan["gaps"]["audio_variants"]
    total_audio = audio["total"]
    print(
        "Audio variants:        "
        f"{total_audio['current']}/{total_audio['target']} "
        f"(need {total_audio['needed']})"
    )
    for group_name in ("tracks", "splits"):
        print(f"Audio {group_name.title()}:")
        for key, item in audio[group_name].items():
            print(
                f"  {key:28s} {item['current']:>4}/{item['target']:<4} "
                f"need {item['needed']}"
            )
    if plan["recommended_next_scenarios"]:
        print("Recommended next scenario batches:")
        for item in plan["recommended_next_scenarios"]:
            print(
                f"  {item['count']:>3} x domain={item['domain']} "
                f"track={item['track']} difficulty={item['difficulty']} "
                f"split={item['split']}"
            )


def _print_run_manifest(manifest: dict[str, Any]) -> None:
    print("\nLATENCY-COST-QUALITY RUN MANIFEST")
    print("=" * 88)
    release = manifest["release_tuple"]
    print(f"Manifest version:      {manifest['manifest_version']}")
    print(f"Seed:                  {release['seed']}")
    env = release.get("environment", {})
    print(f"Region/network:        {env.get('region', 'n/a')} / {env.get('network', 'n/a')}")
    print(f"Hardware profile:      {env.get('hardware_profile', 'n/a')}")
    print(f"Transport:             {env.get('transport', 'n/a')}")
    print(f"Concurrency levels:    {env.get('concurrency_levels')}")
    judge = release.get("judge", {})
    print(f"Judge model:           {judge.get('model') or 'n/a'}")
    print(f"Reports:               {len(manifest['reports'])}")
    print(f"Systems:               {', '.join(system['name'] for system in manifest['systems'])}")
    pricing = release.get("pricing_manifest") or {}
    print(f"Pricing manifest hash: {pricing.get('sha256', 'n/a')}")
    changelog = release.get("changelog") or {}
    print(f"Changelog hash:        {changelog.get('sha256', 'n/a')}")
    baselines = release.get("baseline_manifest") or {}
    print(f"Baseline manifest hash:{baselines.get('sha256', 'n/a'):>64}")


def _print_release_bundle(bundle: dict[str, Any]) -> None:
    print("\nLATENCY-COST-QUALITY RELEASE BUNDLE")
    print("=" * 88)
    print(f"Output:                {bundle['output_dir']}")
    print(f"Readiness profile:     {bundle['release_tuple']['readiness_profile']}")
    env = bundle["release_tuple"]["environment"]
    print(f"Region/network:        {env.get('region', 'n/a')} / {env.get('network', 'n/a')}")
    print(f"Hardware profile:      {env.get('hardware_profile', 'n/a')}")
    print(f"Transport:             {env.get('transport', 'n/a')}")
    print(f"Artifacts:             {len(bundle['artifacts'])}")
    print(f"Pinned input files:    {_count_bundle_inputs(bundle.get('input_files', {}))}")
    print(f"Validation passed:     {bundle['validation']['passed']}")
    if not bundle["validation"]["passed"]:
        print("Issues:")
        for section in ("frontier_report", "run_manifest", "readiness"):
            item = bundle["validation"][section]
            if item["passed"]:
                continue
            print(f"  {section}: {item['num_issues']} issue(s)")


def _print_release_verification(verification: dict[str, Any]) -> None:
    print("\nOPENVOICECS-BENCH RELEASE VERIFICATION")
    print("=" * 88)
    print(f"Passed:                {verification['passed']}")
    print(f"Checks:                {verification['num_checks']}")
    print(f"Issues:                {verification['num_issues']}")
    print(f"Readiness profile:     {verification.get('readiness_profile', 'n/a')}")
    print("Checklist:")
    for check in verification.get("checks", []):
        status = "pass" if check.get("passed") else "fail"
        print(f"  {check.get('name', 'unknown'):24s} {status:4s} {check.get('num_issues', 0)}")
    if verification.get("issues"):
        print("\nIssues:")
        for issue in verification["issues"]:
            print(
                f"  {issue['check']}::{issue['item_id']}::{issue['path']}: "
                f"{issue['message']}"
            )


def _count_bundle_inputs(input_files: dict[str, Any]) -> int:
    if not isinstance(input_files, dict):
        return 0
    count = 0
    reports = input_files.get("reports")
    if isinstance(reports, list):
        count += len(reports)
    count += sum(
        1
        for key, value in input_files.items()
        if key != "reports" and value is not None
    )
    return count


def _print_judge_report(report: dict[str, Any]) -> None:
    print("\nOPENVOICECS-BENCH JUDGE REPORT")
    print("=" * 88)
    print(f"Rubric:                {report['rubric']['name']} v{report['rubric']['version']}")
    print(f"Annotations:           {report['num_annotations']}")
    print(f"Items:                 {report['num_items']}")
    print(f"Raters:                {report['num_raters']}")
    print(f"Subjective score:      {report['overall_subjective_score']:.1%}")
    coverage = report["coverage"]
    print(
        "Coverage:              "
        f"{coverage['items_meeting_minimum_raters']}/{report['num_items']} "
        f"items meet min raters={coverage['minimum_raters_per_item']}"
    )
    agreement = report["agreement"]
    print(f"Agreement alpha:       {_fmt(agreement.get('overall'))}")
    print("\nDimension scores:")
    scale = report["rubric"]["scale"]
    for dimension, score in report["dimension_scores"].items():
        print(f"  {dimension:26s} {score:.2f} / {scale['max']}")


def _print_judged_benchmark_report(report: dict[str, Any]) -> None:
    print("\nOPENVOICECS-BENCH JUDGED REPORT")
    print("=" * 88)
    print(f"Source benchmark:      {report.get('benchmark', 'n/a')}")
    print(f"Scenarios:             {report.get('num_scenarios', 0)}")
    print(f"Deterministic score:   {report.get('overall_score', 0):.2f} / 100")
    print(f"Judged exp score:      {_fmt_pct(report.get('conversation_experience_score'))}")
    experience = report.get("conversation_experience", {})
    print(f"Judged coverage:       {_fmt_pct(experience.get('coverage'))}")
    print(f"Judged trials:         {experience.get('num_judged_trials', 0)}")
    judges = experience.get("judge_counts", {})
    if judges:
        print(f"Judge sources:         {', '.join(judges)}")


def _print_model_judge_result(
    *,
    annotations: list[dict[str, Any]],
    annotations_output: Path,
    judge_report: dict[str, Any],
    judge_report_output: Path,
    judged_report: dict[str, Any] | None,
    judged_report_output: Path | None,
) -> None:
    print("\nOPENVOICECS-BENCH MODEL JUDGE")
    print("=" * 88)
    print(f"Annotations:           {len(annotations)}")
    print(f"Items:                 {judge_report.get('num_items', 0)}")
    print(f"Raters:                {judge_report.get('num_raters', 0)}")
    print(f"Subjective score:      {judge_report.get('overall_subjective_score', 0):.1%}")
    coverage = judge_report.get("coverage", {})
    print(
        "Coverage:              "
        f"{coverage.get('items_meeting_minimum_raters', 0)}/"
        f"{judge_report.get('num_items', 0)} items meet min raters"
    )
    print(f"Annotations output:    {annotations_output}")
    print(f"Judge report output:   {judge_report_output}")
    if judged_report is not None and judged_report_output is not None:
        judged_score = _fmt_pct(judged_report.get("conversation_experience_score"))
        print(f"Judged exp score:      {judged_score}")
        print(f"Judged report output:  {judged_report_output}")


def _print_submission_card(card: dict[str, Any]) -> None:
    print("\nOPENVOICECS-BENCH SUBMISSION CARD")
    print("=" * 88)
    system = card["system"]
    evaluation = card["evaluation"]
    pricing = card["pricing"]
    print(f"System:                {system['name']}")
    print(f"Provider:              {system.get('provider')}")
    print(f"Model ID:              {system.get('model_id') or 'n/a'}")
    print(f"Submission spec:       {system.get('submission_spec') or 'n/a'}")
    print(f"Pipeline/input:        {system.get('pipeline_type')} / {system.get('input_modality')}")
    print(f"Pricing profile:       {pricing.get('pricing_profile_id') or 'n/a'}")
    print(f"Report hash:           {(evaluation.get('report') or {}).get('sha256', 'n/a')}")
    print(f"Score:                 {evaluation.get('overall_score')}")


def _parse_utility_weights(value: str) -> dict[str, float]:
    weights = {}
    for item in value.split(","):
        key, raw = item.split("=", 1)
        weights[key.strip()] = float(raw)
    return weights


def _parse_headers(values: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError("headers must use KEY=VALUE syntax")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("header key cannot be empty")
        headers[key] = value
    return headers


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value}{suffix}"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OpenVoiceCS-Bench")
    parser.add_argument(
        "--scenarios",
        default=str(DEFAULT_SCENARIO_PATH),
        help="Path to OpenVoiceCS scenario JSON",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate and summarize scenarios")
    validate.add_argument(
        "--audio-manifest",
        default=str(DEFAULT_AUDIO_MANIFEST_PATH),
        help="Optional audio manifest to validate",
    )
    validate.set_defaults(func=cmd_validate)

    validate_report = subparsers.add_parser(
        "validate-report",
        help="Validate saved report JSON before leaderboard/frontier ingestion",
    )
    validate_report.add_argument("reports", nargs="+", help="Saved report JSON files")
    validate_report.set_defaults(func=cmd_validate_report)

    validate_frontier = subparsers.add_parser(
        "validate-frontier",
        help="Validate saved latency-cost-quality frontier report JSON",
    )
    validate_frontier.add_argument("reports", nargs="+", help="Saved frontier JSON files")
    validate_frontier.set_defaults(func=cmd_validate_frontier)

    validate_run_manifest = subparsers.add_parser(
        "validate-run-manifest",
        help="Validate a frozen run manifest JSON",
    )
    validate_run_manifest.add_argument("manifest", help="Saved run manifest JSON")
    validate_run_manifest.set_defaults(func=cmd_validate_run_manifest)

    validate_release_bundle = subparsers.add_parser(
        "validate-release-bundle",
        help="Validate a saved frontier release bundle and artifact hashes",
    )
    validate_release_bundle.add_argument(
        "bundles",
        nargs="+",
        help="Saved release_bundle.json files",
    )
    validate_release_bundle.set_defaults(func=cmd_validate_release_bundle)

    verify_release = subparsers.add_parser(
        "verify-release",
        help="Run all OpenVoiceCS release validators and release gates",
    )
    verify_release.add_argument("--audio-manifest", default=str(DEFAULT_AUDIO_MANIFEST_PATH))
    verify_release.add_argument("--audio-root", default=".")
    verify_release.add_argument(
        "--require-audio-assets",
        action="store_true",
        help="Require referenced WAV assets to exist and match pinned metadata",
    )
    verify_release.add_argument("--pricing-manifest", default=str(DEFAULT_PRICING_MANIFEST_PATH))
    verify_release.add_argument("--splits", default=str(DEFAULT_SPLIT_MANIFEST_PATH))
    verify_release.add_argument("--split-commitments", default=str(DEFAULT_SPLIT_COMMITMENT_PATH))
    verify_release.add_argument("--provenance", default=str(DEFAULT_PROVENANCE_MANIFEST_PATH))
    verify_release.add_argument("--changelog", default=str(DEFAULT_CHANGELOG_PATH))
    verify_release.add_argument("--baseline-manifest", default=str(DEFAULT_BASELINE_MANIFEST_PATH))
    verify_release.add_argument("--review-manifest", default=str(DEFAULT_REVIEW_MANIFEST_PATH))
    verify_release.add_argument("--datasheet", default=str(DEFAULT_DATASHEET_PATH))
    verify_release.add_argument("--judge-protocol", default=str(DEFAULT_JUDGE_PROTOCOL_PATH))
    verify_release.add_argument(
        "--judge-annotation-package",
        default=str(DEFAULT_JUDGE_ANNOTATION_PACKAGE_PATH),
    )
    verify_release.add_argument("--sealed-ops", default=str(DEFAULT_SEALED_OPS_PATH))
    verify_release.add_argument("--judge-study", default=str(DEFAULT_JUDGE_STUDY_PATH))
    verify_release.add_argument("--sealed-queue", default=str(DEFAULT_SEALED_QUEUE_PATH))
    verify_release.add_argument(
        "--external-endpoint-contract",
        default=str(DEFAULT_EXTERNAL_ENDPOINT_CONTRACT_PATH),
    )
    verify_release.add_argument("--external-systems", default=str(DEFAULT_EXTERNAL_SYSTEMS_PATH))
    verify_release.add_argument("--claims", default=str(DEFAULT_CLAIMS_MANIFEST_PATH))
    verify_release.add_argument("--submission-intake", default=str(DEFAULT_SUBMISSION_INTAKE_PATH))
    verify_release.add_argument("--release-audit", default=str(DEFAULT_RELEASE_AUDIT_PATH))
    verify_release.add_argument(
        "--readiness-profile",
        choices=sorted(RELEASE_PROFILES),
        default="seed",
        help="Readiness profile to include in release verification",
    )
    verify_release.add_argument("--frontier-report", default=None)
    verify_release.add_argument("--run-manifest", default=None)
    verify_release.add_argument("--plot-dir", default=None)
    verify_release.add_argument("--output", default=None)
    verify_release.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when any release verification check fails",
    )
    verify_release.set_defaults(func=cmd_verify_release)

    validate_judge_rubric = subparsers.add_parser(
        "validate-judge-rubric",
        help="Validate a subjective-quality judge rubric JSON",
    )
    validate_judge_rubric.add_argument(
        "--rubric",
        default=str(DEFAULT_JUDGE_RUBRIC_PATH),
        help="Judge rubric JSON path",
    )
    validate_judge_rubric.set_defaults(func=cmd_validate_judge_rubric)

    validate_judge_protocol = subparsers.add_parser(
        "validate-judge-protocol",
        help="Validate the subjective-quality judge protocol JSON",
    )
    validate_judge_protocol.add_argument(
        "--protocol",
        default=str(DEFAULT_JUDGE_PROTOCOL_PATH),
        help="Judge protocol JSON path",
    )
    validate_judge_protocol.set_defaults(func=cmd_validate_judge_protocol)

    validate_judge_study = subparsers.add_parser(
        "validate-judge-study",
        help="Validate human/audited model-judge study design and evidence",
    )
    validate_judge_study.add_argument(
        "--study",
        default=str(DEFAULT_JUDGE_STUDY_PATH),
        help="Judge study JSON path",
    )
    validate_judge_study.set_defaults(func=cmd_validate_judge_study)

    validate_judge_annotation_package = subparsers.add_parser(
        "validate-judge-annotation-package",
        help="Validate a judge annotation package and referenced evidence",
    )
    validate_judge_annotation_package.add_argument(
        "--package",
        default=str(DEFAULT_JUDGE_ANNOTATION_PACKAGE_PATH),
        help="Judge annotation package JSON path",
    )
    validate_judge_annotation_package.set_defaults(
        func=cmd_validate_judge_annotation_package
    )

    validate_judge_report = subparsers.add_parser(
        "validate-judge-report",
        help="Validate aggregated judge reports and release-quality gates",
    )
    validate_judge_report.add_argument(
        "reports",
        nargs="+",
        help="Saved judge report JSON files",
    )
    validate_judge_report.set_defaults(func=cmd_validate_judge_report)

    validate_sealed_ops = subparsers.add_parser(
        "validate-sealed-ops",
        help="Validate sealed-test operations policy and split privacy evidence",
    )
    validate_sealed_ops.add_argument(
        "--sealed-ops",
        default=str(DEFAULT_SEALED_OPS_PATH),
        help="Sealed operations JSON path",
    )
    validate_sealed_ops.add_argument("--splits", default=str(DEFAULT_SPLIT_MANIFEST_PATH))
    validate_sealed_ops.add_argument(
        "--split-commitments",
        default=str(DEFAULT_SPLIT_COMMITMENT_PATH),
    )
    validate_sealed_ops.set_defaults(func=cmd_validate_sealed_ops)

    validate_sealed_queue = subparsers.add_parser(
        "validate-sealed-queue",
        help="Validate hosted sealed-evaluator queue attempts and audit controls",
    )
    validate_sealed_queue.add_argument(
        "--queue",
        default=str(DEFAULT_SEALED_QUEUE_PATH),
        help="Sealed evaluator queue JSON path",
    )
    validate_sealed_queue.add_argument(
        "--sealed-ops",
        default=str(DEFAULT_SEALED_OPS_PATH),
        help="Sealed operations JSON path",
    )
    validate_sealed_queue.add_argument(
        "--split-commitments",
        default=str(DEFAULT_SPLIT_COMMITMENT_PATH),
    )
    validate_sealed_queue.set_defaults(func=cmd_validate_sealed_queue)

    validate_external_systems = subparsers.add_parser(
        "validate-external-systems",
        help="Validate hash-pinned external-system registry evidence",
    )
    validate_external_systems.add_argument(
        "--registry",
        default=str(DEFAULT_EXTERNAL_SYSTEMS_PATH),
        help="External systems registry JSON path",
    )
    validate_external_systems.set_defaults(func=cmd_validate_external_systems)

    validate_external_endpoint_contract = subparsers.add_parser(
        "validate-external-endpoint-contract",
        help="Validate the provider-neutral external endpoint protocol contract",
    )
    validate_external_endpoint_contract.add_argument(
        "--contract",
        default=str(DEFAULT_EXTERNAL_ENDPOINT_CONTRACT_PATH),
        help="External endpoint contract JSON path",
    )
    validate_external_endpoint_contract.set_defaults(
        func=cmd_validate_external_endpoint_contract
    )

    validate_claims = subparsers.add_parser(
        "validate-claims",
        help="Validate leaderboard claim package and comparison evidence",
    )
    validate_claims.add_argument(
        "--claims",
        default=str(DEFAULT_CLAIMS_MANIFEST_PATH),
        help="Leaderboard claims JSON path",
    )
    validate_claims.set_defaults(func=cmd_validate_claims)

    validate_submission_intake = subparsers.add_parser(
        "validate-submission-intake",
        help="Validate an official submission intake envelope and hash-pinned artifacts",
    )
    validate_submission_intake.add_argument(
        "--intake",
        default=str(DEFAULT_SUBMISSION_INTAKE_PATH),
        help="Submission intake JSON path",
    )
    validate_submission_intake.set_defaults(func=cmd_validate_submission_intake)

    validate_splits = subparsers.add_parser(
        "validate-splits",
        help="Validate public/sealed split manifest assignments",
    )
    validate_splits.add_argument("--splits", default=str(DEFAULT_SPLIT_MANIFEST_PATH))
    validate_splits.add_argument("--audio-manifest", default=str(DEFAULT_AUDIO_MANIFEST_PATH))
    validate_splits.set_defaults(func=cmd_validate_splits)

    split_commitments = subparsers.add_parser(
        "split-commitments",
        help="Build cryptographic commitments for public/sealed split contents",
    )
    split_commitments.add_argument("--splits", default=str(DEFAULT_SPLIT_MANIFEST_PATH))
    split_commitments.add_argument("--audio-manifest", default=str(DEFAULT_AUDIO_MANIFEST_PATH))
    split_commitments.add_argument("--output", default=str(DEFAULT_SPLIT_COMMITMENT_PATH))
    split_commitments.add_argument(
        "--hide-public-ids",
        action="store_true",
        help="Do not reveal public_dev item IDs in the commitment file",
    )
    split_commitments.add_argument(
        "--reveal-sealed-ids",
        action="store_true",
        help="Reveal sealed_test item IDs; normally leave disabled for public commitments",
    )
    split_commitments.set_defaults(func=cmd_split_commitments)

    validate_split_commitments = subparsers.add_parser(
        "validate-split-commitments",
        help="Validate split commitments against scenario/audio/split files",
    )
    validate_split_commitments.add_argument(
        "commitments",
        nargs="?",
        default=str(DEFAULT_SPLIT_COMMITMENT_PATH),
        help="Saved split commitment JSON path",
    )
    validate_split_commitments.add_argument("--splits", default=str(DEFAULT_SPLIT_MANIFEST_PATH))
    validate_split_commitments.add_argument(
        "--audio-manifest",
        default=str(DEFAULT_AUDIO_MANIFEST_PATH),
    )
    validate_split_commitments.set_defaults(func=cmd_validate_split_commitments)

    datasheet = subparsers.add_parser(
        "datasheet",
        help="Build a machine-readable benchmark datasheet for release governance",
    )
    datasheet.add_argument("--audio-manifest", default=str(DEFAULT_AUDIO_MANIFEST_PATH))
    datasheet.add_argument("--pricing-manifest", default=str(DEFAULT_PRICING_MANIFEST_PATH))
    datasheet.add_argument("--splits", default=str(DEFAULT_SPLIT_MANIFEST_PATH))
    datasheet.add_argument("--provenance", default=str(DEFAULT_PROVENANCE_MANIFEST_PATH))
    datasheet.add_argument("--changelog", default=str(DEFAULT_CHANGELOG_PATH))
    datasheet.add_argument("--baseline-manifest", default=str(DEFAULT_BASELINE_MANIFEST_PATH))
    datasheet.add_argument("--review-manifest", default=str(DEFAULT_REVIEW_MANIFEST_PATH))
    datasheet.add_argument(
        "--split-commitments",
        default=str(DEFAULT_SPLIT_COMMITMENT_PATH),
        help="Optional split commitment JSON path",
    )
    datasheet.add_argument("--output", default=str(DEFAULT_DATASHEET_PATH))
    datasheet.set_defaults(func=cmd_datasheet)

    validate_datasheet = subparsers.add_parser(
        "validate-datasheet",
        help="Validate saved benchmark datasheet JSON files",
    )
    validate_datasheet.add_argument(
        "datasheets",
        nargs="+",
        help="Saved benchmark datasheet JSON files",
    )
    validate_datasheet.set_defaults(func=cmd_validate_datasheet)

    baselines = subparsers.add_parser(
        "baselines",
        help="Generate reproducible oracle/no-op reference baseline reports",
    )
    baselines.add_argument("--audio-manifest", default=str(DEFAULT_AUDIO_MANIFEST_PATH))
    baselines.add_argument("--trials", type=int, default=3)
    baselines.add_argument("--output-dir", default=str(DEFAULT_BASELINE_DIR))
    baselines.add_argument("--output", default=str(DEFAULT_BASELINE_MANIFEST_PATH))
    baselines.set_defaults(func=cmd_baselines)

    validate_baselines = subparsers.add_parser(
        "validate-baselines",
        help="Validate reference baseline manifests and report hashes",
    )
    validate_baselines.add_argument(
        "manifests",
        nargs="+",
        help="Saved reference baseline manifest JSON files",
    )
    validate_baselines.set_defaults(func=cmd_validate_baselines)

    validate_audio_assets = subparsers.add_parser(
        "validate-audio-assets",
        help="Validate audio asset files, hashes, sample rates, and durations",
    )
    validate_audio_assets.add_argument(
        "--audio-manifest",
        default=str(DEFAULT_AUDIO_MANIFEST_PATH),
    )
    validate_audio_assets.add_argument("--audio-root", default=".")
    validate_audio_assets.add_argument(
        "--allow-missing-sha256",
        action="store_true",
        help="Validate files and WAV metadata without requiring pinned hashes",
    )
    validate_audio_assets.add_argument(
        "--duration-tolerance-seconds",
        type=float,
        default=0.05,
    )
    validate_audio_assets.set_defaults(func=cmd_validate_audio_assets)

    validate_provenance = subparsers.add_parser(
        "validate-provenance",
        help="Validate scenario/audio provenance, consent, and contamination metadata",
    )
    validate_provenance.add_argument(
        "--provenance",
        default=str(DEFAULT_PROVENANCE_MANIFEST_PATH),
    )
    validate_provenance.add_argument(
        "--audio-manifest",
        default=str(DEFAULT_AUDIO_MANIFEST_PATH),
    )
    validate_provenance.set_defaults(func=cmd_validate_provenance)

    validate_changelog = subparsers.add_parser(
        "validate-changelog",
        help="Validate release changelog and errata metadata",
    )
    validate_changelog.add_argument(
        "--changelog",
        default=str(DEFAULT_CHANGELOG_PATH),
    )
    validate_changelog.add_argument(
        "--audio-manifest",
        default=str(DEFAULT_AUDIO_MANIFEST_PATH),
    )
    validate_changelog.set_defaults(func=cmd_validate_changelog)

    validate_reviews = subparsers.add_parser(
        "validate-reviews",
        help="Validate scenario review approval metadata",
    )
    validate_reviews.add_argument(
        "--review-manifest",
        default=str(DEFAULT_REVIEW_MANIFEST_PATH),
    )
    validate_reviews.set_defaults(func=cmd_validate_reviews)

    pin_audio_assets = subparsers.add_parser(
        "pin-audio-assets",
        help="Compute SHA-256, sample rate, and duration metadata for audio assets",
    )
    pin_audio_assets.add_argument(
        "--audio-manifest",
        default=str(DEFAULT_AUDIO_MANIFEST_PATH),
    )
    pin_audio_assets.add_argument("--audio-root", default=".")
    pin_audio_assets.add_argument("--output", default=None)
    pin_audio_assets.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite the input manifest in place",
    )
    pin_audio_assets.set_defaults(func=cmd_pin_audio_assets)

    add_scenarios = subparsers.add_parser(
        "add-scenarios",
        help="Append validated scenario drafts and update split/provenance outputs",
    )
    add_scenarios.add_argument("draft", help="Scenario object or scenario-suite JSON")
    add_scenarios.add_argument("--audio-manifest", default=str(DEFAULT_AUDIO_MANIFEST_PATH))
    add_scenarios.add_argument("--splits", default=str(DEFAULT_SPLIT_MANIFEST_PATH))
    add_scenarios.add_argument("--provenance", default=str(DEFAULT_PROVENANCE_MANIFEST_PATH))
    add_scenarios.add_argument("--split", default="public_dev")
    add_scenarios.add_argument("--source-type", default="hand_authored_synthetic")
    add_scenarios.add_argument("--license", default="CC-BY-4.0")
    add_scenarios.add_argument(
        "--authoring-method",
        default="curated benchmark expansion",
    )
    add_scenarios.add_argument("--contamination-risk", default="low")
    add_scenarios.add_argument("--review-status", default="draft")
    add_scenarios.add_argument("--output-scenarios", required=True)
    add_scenarios.add_argument("--output-splits", required=True)
    add_scenarios.add_argument("--output-provenance", required=True)
    add_scenarios.set_defaults(func=cmd_add_scenarios)

    scaffold_scenarios = subparsers.add_parser(
        "scaffold-scenarios",
        help="Create incomplete scenario draft skeletons from coverage gaps",
    )
    scaffold_scenarios.add_argument("--splits", default=str(DEFAULT_SPLIT_MANIFEST_PATH))
    scaffold_scenarios.add_argument("--targets", default=str(DEFAULT_COVERAGE_TARGET_PATH))
    scaffold_scenarios.add_argument("--profile", default="public_beta")
    scaffold_scenarios.add_argument(
        "--count",
        type=int,
        default=None,
        help="Limit the number of draft skeletons to write",
    )
    scaffold_scenarios.add_argument("--output", required=True)
    scaffold_scenarios.set_defaults(func=cmd_scaffold_scenarios)

    coverage_plan = subparsers.add_parser(
        "coverage-plan",
        help="Compare current scenario coverage against target release profiles",
    )
    coverage_plan.add_argument("--splits", default=str(DEFAULT_SPLIT_MANIFEST_PATH))
    coverage_plan.add_argument("--audio-manifest", default=str(DEFAULT_AUDIO_MANIFEST_PATH))
    coverage_plan.add_argument("--targets", default=str(DEFAULT_COVERAGE_TARGET_PATH))
    coverage_plan.add_argument("--profile", default="public_beta")
    coverage_plan.add_argument("--output", default=None)
    coverage_plan.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when coverage targets are not met",
    )
    coverage_plan.set_defaults(func=cmd_coverage_plan)

    judge_report = subparsers.add_parser(
        "judge-report",
        help="Aggregate human/model judge annotations for a saved report",
    )
    judge_report.add_argument("report", help="Source OpenVoiceCS report JSON")
    judge_report.add_argument("annotations", help="Judge annotations JSONL or JSON")
    judge_report.add_argument("--rubric", default=str(DEFAULT_JUDGE_RUBRIC_PATH))
    judge_report.add_argument("--output", default=None)
    judge_report.set_defaults(func=cmd_judge_report)

    model_judge = subparsers.add_parser(
        "model-judge",
        help="Call audited model judges, write annotations, and aggregate judged reports",
    )
    model_judge.add_argument("report", help="Source OpenVoiceCS report JSON")
    model_judge.add_argument(
        "--judge",
        action="append",
        required=True,
        help="Judge spec as provider:model_id, repeat for two or more judges",
    )
    model_judge.add_argument(
        "--adjudicator",
        default=None,
        help="Optional tie-breaker judge spec as provider:model_id",
    )
    model_judge.add_argument("--rubric", default=str(DEFAULT_JUDGE_RUBRIC_PATH))
    model_judge.add_argument(
        "--prompt",
        default="data/openvoicecs/judging/judge_prompt_v0.1.md",
    )
    model_judge.add_argument("--judge-protocol", default=str(DEFAULT_JUDGE_PROTOCOL_PATH))
    model_judge.add_argument("--annotations-output", required=True)
    model_judge.add_argument("--judge-report-output", required=True)
    model_judge.add_argument("--judged-report-output", default=None)
    model_judge.add_argument("--disagreement-threshold", type=float, default=None)
    model_judge.add_argument("--max-output-tokens", type=int, default=700)
    model_judge.add_argument("--temperature", type=float, default=0.0)
    model_judge.add_argument("--timeout-seconds", type=float, default=60.0)
    model_judge.add_argument(
        "--env",
        default=".env",
        help="Environment file with provider API keys",
    )
    model_judge.set_defaults(func=cmd_model_judge)

    apply_judge = subparsers.add_parser(
        "apply-judge-report",
        help="Attach an aggregated judge report to a benchmark report",
    )
    apply_judge.add_argument("report", help="Source OpenVoiceCS report JSON")
    apply_judge.add_argument("judge_report", help="Aggregated judge report JSON")
    apply_judge.add_argument("--output", default=None)
    apply_judge.set_defaults(func=cmd_apply_judge_report)

    compare = subparsers.add_parser(
        "compare",
        help="Compare two saved reports with paired CIs and McNemar exact test",
    )
    compare.add_argument("baseline", help="Baseline report JSON")
    compare.add_argument("candidate", help="Candidate report JSON")
    compare.add_argument("--iterations", type=int, default=10000)
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--confidence", type=float, default=0.95)
    compare.add_argument("--output", default=None)
    compare.set_defaults(func=cmd_compare)

    score = subparsers.add_parser("score", help="Score a built-in baseline agent")
    score.add_argument("--agent", choices=["oracle", "noop"], default="oracle")
    score.add_argument("--trials", type=int, default=1)
    score.add_argument("--max", type=int, default=None)
    score.add_argument("--track", default=None)
    score.add_argument("--output", default=None)
    score.set_defaults(func=cmd_score)

    score_audio = subparsers.add_parser(
        "score-audio",
        help="Score a built-in baseline agent over audio manifest variants",
    )
    score_audio.add_argument("--agent", choices=["oracle", "noop"], default="oracle")
    score_audio.add_argument("--trials", type=int, default=1)
    score_audio.add_argument("--max", type=int, default=None)
    score_audio.add_argument("--track", default=None, help="audio_to_action, robustness, etc.")
    score_audio.add_argument("--audio-manifest", default=str(DEFAULT_AUDIO_MANIFEST_PATH))
    score_audio.add_argument("--output", default=None)
    score_audio.set_defaults(func=cmd_score_audio)

    submit = subparsers.add_parser("submit", help="Score an external Python submission adapter")
    submit.add_argument("submission", help="Adapter spec in the form path.py:function")
    submit.add_argument("--mode", choices=["text", "audio"], default="text")
    submit.add_argument("--name", default=None, help="Display name for report metadata")
    submit.add_argument("--trials", type=int, default=1)
    submit.add_argument("--max", type=int, default=None)
    submit.add_argument("--track", default=None)
    submit.add_argument("--audio-manifest", default=str(DEFAULT_AUDIO_MANIFEST_PATH))
    submit.add_argument("--provider", default=None)
    submit.add_argument("--model-id", default=None)
    submit.add_argument("--pricing-profile-id", default=None)
    submit.add_argument("--pricing-snapshot-date", default=None)
    submit.add_argument(
        "--pipeline-type",
        choices=["cascaded", "native_speech_to_speech", "unknown"],
        default="unknown",
    )
    submit.add_argument("--output", default=None)
    submit.set_defaults(func=cmd_submit)

    score_provider = subparsers.add_parser(
        "score-provider",
        help="Score a hosted model provider through the OpenVoiceCS trace adapter",
    )
    score_provider.add_argument(
        "--provider",
        required=True,
        choices=sorted(PIPELINE_PROVIDERS),
        help="Provider adapter to use",
    )
    score_provider.add_argument(
        "--model",
        default=None,
        help=f"Model ID. Defaults by provider: {DEFAULT_MODEL_IDS}",
    )
    score_provider.add_argument("--mode", choices=["text", "audio"], default="text")
    score_provider.add_argument("--name", default=None, help="Display name for report metadata")
    score_provider.add_argument("--trials", type=int, default=1)
    score_provider.add_argument("--max", type=int, default=None)
    score_provider.add_argument("--track", default=None)
    score_provider.add_argument("--audio-manifest", default=str(DEFAULT_AUDIO_MANIFEST_PATH))
    score_provider.add_argument("--api-key", default=None, help="Explicit API key; env vars are preferred")
    score_provider.add_argument("--base-url", default=None, help="Override OpenAI-compatible base URL")
    score_provider.add_argument("--temperature", type=float, default=0.1)
    score_provider.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default=None,
        help="OpenAI reasoning effort for reasoning-capable models",
    )
    score_provider.add_argument(
        "--native-tools",
        dest="native_tools",
        action="store_true",
        default=None,
        help="Use native tool calls and execute scenario tools before final JSON",
    )
    score_provider.add_argument(
        "--json-trace",
        dest="native_tools",
        action="store_false",
        help="Use the JSON action-loop fallback instead of native tools",
    )
    score_provider.add_argument("--max-output-tokens", type=int, default=700)
    score_provider.add_argument(
        "--input-price",
        type=float,
        default=None,
        help="USD per 1M input tokens for cost estimates",
    )
    score_provider.add_argument(
        "--output-price",
        type=float,
        default=None,
        help="USD per 1M output tokens for cost estimates",
    )
    score_provider.add_argument("--pricing-profile-id", default=None)
    score_provider.add_argument("--pricing-snapshot-date", default=None)
    score_provider.add_argument("--output", default=None)
    score_provider.set_defaults(func=cmd_score_provider)

    submit_endpoint = subparsers.add_parser(
        "submit-endpoint",
        help="Score a deployed OpenVoiceCS HTTP JSON endpoint submission",
    )
    submit_endpoint.add_argument("endpoint", help="HTTP endpoint URL")
    submit_endpoint.add_argument("--mode", choices=["text", "audio"], default="text")
    submit_endpoint.add_argument("--name", default=None, help="Display name for report metadata")
    submit_endpoint.add_argument("--trials", type=int, default=1)
    submit_endpoint.add_argument("--max", type=int, default=None)
    submit_endpoint.add_argument("--track", default=None)
    submit_endpoint.add_argument("--audio-manifest", default=str(DEFAULT_AUDIO_MANIFEST_PATH))
    submit_endpoint.add_argument("--timeout-seconds", type=float, default=30.0)
    submit_endpoint.add_argument("--run-id", default=None)
    submit_endpoint.add_argument(
        "--header",
        action="append",
        default=None,
        help="HTTP header as KEY=VALUE; repeat for Authorization or vendor headers",
    )
    submit_endpoint.add_argument("--provider", default=None)
    submit_endpoint.add_argument("--model-id", default=None)
    submit_endpoint.add_argument("--pricing-profile-id", default=None)
    submit_endpoint.add_argument("--pricing-snapshot-date", default=None)
    submit_endpoint.add_argument(
        "--pipeline-type",
        choices=["cascaded", "native_speech_to_speech", "unknown"],
        default="unknown",
    )
    submit_endpoint.add_argument("--output", default=None)
    submit_endpoint.set_defaults(func=cmd_submit_endpoint)

    submission_card = subparsers.add_parser(
        "submission-card",
        help="Build a machine-readable submission disclosure card from a report",
    )
    submission_card.add_argument("report", help="Saved OpenVoiceCS report JSON")
    submission_card.add_argument("--output", default=None)
    submission_card.add_argument("--submitter", default=None)
    submission_card.add_argument("--organization", default=None)
    submission_card.add_argument("--contact", default=None)
    submission_card.add_argument("--repository-url", default=None)
    submission_card.add_argument("--license", default=None)
    submission_card.add_argument("--training-data-statement", default="not_provided")
    submission_card.add_argument("--safety-statement", default="not_provided")
    submission_card.add_argument("--limitations", nargs="*", default=None)
    submission_card.set_defaults(func=cmd_submission_card)

    validate_submission_card = subparsers.add_parser(
        "validate-submission-card",
        help="Validate saved submission disclosure cards",
    )
    validate_submission_card.add_argument(
        "cards",
        nargs="+",
        help="Saved submission card JSON files",
    )
    validate_submission_card.set_defaults(func=cmd_validate_submission_card)

    init_submission = subparsers.add_parser(
        "init-submission",
        help="Write a starter external submission adapter",
    )
    init_submission.add_argument("output", help="Path for the new adapter .py file")
    init_submission.add_argument("--function-name", default="run")
    init_submission.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists",
    )
    init_submission.set_defaults(func=cmd_init_submission)

    leaderboard = subparsers.add_parser(
        "leaderboard",
        help="Build leaderboard from report JSON files",
    )
    leaderboard.add_argument("reports", nargs="+", help="Report JSON paths or glob patterns")
    leaderboard.add_argument("--output", default=None)
    leaderboard.set_defaults(func=cmd_leaderboard)

    frontier = subparsers.add_parser(
        "frontier",
        help="Build latency-cost-quality frontier from reports",
    )
    frontier.add_argument("reports", nargs="+", help="Report JSON paths or glob patterns")
    frontier.add_argument("--output", default=None)
    frontier.add_argument(
        "--plot-dir",
        default=None,
        help="Directory for per-domain 3D/2D SVG plots and plot-data JSON",
    )
    frontier.add_argument(
        "--scorecard-dir",
        default=None,
        help="Directory for standardized scorecard JSON/CSV/Markdown artifacts",
    )
    frontier.add_argument(
        "--pricing-snapshot-date",
        default=None,
        help="Fallback provider pricing snapshot date, e.g. 2026-06-11",
    )
    frontier.add_argument(
        "--pricing-manifest",
        default=str(DEFAULT_PRICING_MANIFEST_PATH),
        help="Pinned pricing manifest used to resolve report pricing profiles",
    )
    frontier.add_argument("--region", default=None, help="Test region/environment label")
    frontier.add_argument("--network", default=None, help="Network condition label")
    frontier.add_argument("--hardware-profile", default=None, help="Controlled client hardware profile label")
    frontier.add_argument(
        "--transport",
        default=None,
        help="Transport label, e.g. websocket or webrtc",
    )
    frontier.add_argument(
        "--concurrency-levels",
        nargs="*",
        type=int,
        default=[1, 10, 100],
        help="Concurrency levels represented in the report set",
    )
    frontier.add_argument("--experience-gate", type=float, default=0.6)
    frontier.add_argument(
        "--latency-target-ms",
        nargs="*",
        type=float,
        default=None,
        help="Optional p95 TTFB latency budgets for constrained frontiers",
    )
    frontier.add_argument(
        "--cost-target-usd",
        nargs="*",
        type=float,
        default=None,
        help="Optional cost-per-successful-conversation budgets for constrained frontiers",
    )
    frontier.add_argument(
        "--utility",
        default=None,
        help="Optional weights such as quality=1,latency=0.2,cost=0.1",
    )
    frontier.set_defaults(func=cmd_frontier)

    release_bundle = subparsers.add_parser(
        "release-bundle",
        help="Build a frozen frontier report, plots, run manifest, and readiness bundle",
    )
    release_bundle.add_argument("reports", nargs="+", help="Report JSON paths or glob patterns")
    release_bundle.add_argument("--output-dir", required=True)
    release_bundle.add_argument("--audio-manifest", default=str(DEFAULT_AUDIO_MANIFEST_PATH))
    release_bundle.add_argument("--audio-root", default=".")
    release_bundle.add_argument("--pricing-manifest", default=str(DEFAULT_PRICING_MANIFEST_PATH))
    release_bundle.add_argument("--splits", default=str(DEFAULT_SPLIT_MANIFEST_PATH))
    release_bundle.add_argument("--provenance", default=str(DEFAULT_PROVENANCE_MANIFEST_PATH))
    release_bundle.add_argument("--changelog", default=str(DEFAULT_CHANGELOG_PATH))
    release_bundle.add_argument("--baseline-manifest", default=str(DEFAULT_BASELINE_MANIFEST_PATH))
    release_bundle.add_argument("--review-manifest", default=str(DEFAULT_REVIEW_MANIFEST_PATH))
    release_bundle.add_argument("--judge-model", default=None)
    release_bundle.add_argument("--judge-prompt", default=None)
    release_bundle.add_argument("--seed", type=int, default=0)
    release_bundle.add_argument("--region", default=None)
    release_bundle.add_argument("--network", default=None)
    release_bundle.add_argument("--hardware-profile", default=None)
    release_bundle.add_argument("--transport", default=None)
    release_bundle.add_argument(
        "--concurrency-levels",
        nargs="*",
        type=int,
        default=[1, 10, 100],
    )
    release_bundle.add_argument("--pricing-snapshot-date", default=None)
    release_bundle.add_argument("--experience-gate", type=float, default=0.6)
    release_bundle.add_argument(
        "--latency-target-ms",
        nargs="*",
        type=float,
        default=None,
    )
    release_bundle.add_argument(
        "--cost-target-usd",
        nargs="*",
        type=float,
        default=None,
    )
    release_bundle.add_argument(
        "--utility",
        default=None,
        help="Optional weights such as quality=1,latency=0.2,cost=0.1",
    )
    release_bundle.add_argument(
        "--readiness-profile",
        choices=sorted(RELEASE_PROFILES),
        default="seed",
    )
    release_bundle.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when bundle validation or readiness does not pass",
    )
    release_bundle.set_defaults(func=cmd_release_bundle)

    load = subparsers.add_parser(
        "load",
        help="Run the reference realtime client under controlled concurrency",
    )
    load.add_argument("--agent", choices=["oracle", "noop"], default="oracle")
    load.add_argument(
        "--endpoint",
        default=None,
        help="Optional WebSocket endpoint or WebRTC signaling endpoint",
    )
    load.add_argument(
        "--transport",
        choices=["in_process", "websocket", "webrtc"],
        default=None,
        help="Transport to evaluate; defaults to websocket when --endpoint is set",
    )
    load.add_argument("--timeout-seconds", type=float, default=30.0)
    load.add_argument("--trials", type=int, default=1)
    load.add_argument("--max", type=int, default=None)
    load.add_argument("--track", default=None)
    load.add_argument(
        "--concurrency-levels",
        nargs="+",
        type=int,
        default=[1, 10, 100],
        help="Concurrent calls to run, e.g. 1 10 100",
    )
    load.add_argument("--region", default="unspecified")
    load.add_argument("--network", default="unspecified")
    load.add_argument("--hardware-profile", default="unspecified")
    load.add_argument("--seed", type=int, default=0)
    load.add_argument("--pricing-snapshot-date", default=None)
    load.add_argument("--output", default=None)
    load.set_defaults(func=cmd_load)

    audit = subparsers.add_parser(
        "audit",
        help="Build release audit for scenario and audio manifests",
    )
    audit.add_argument("--audio-manifest", default=str(DEFAULT_AUDIO_MANIFEST_PATH))
    audit.add_argument("--audio-root", default=".")
    audit.add_argument("--pricing-manifest", default=str(DEFAULT_PRICING_MANIFEST_PATH))
    audit.add_argument("--splits", default=str(DEFAULT_SPLIT_MANIFEST_PATH))
    audit.add_argument("--split-commitments", default=str(DEFAULT_SPLIT_COMMITMENT_PATH))
    audit.add_argument("--provenance", default=str(DEFAULT_PROVENANCE_MANIFEST_PATH))
    audit.add_argument("--changelog", default=str(DEFAULT_CHANGELOG_PATH))
    audit.add_argument("--baseline-manifest", default=str(DEFAULT_BASELINE_MANIFEST_PATH))
    audit.add_argument("--review-manifest", default=str(DEFAULT_REVIEW_MANIFEST_PATH))
    audit.add_argument("--judge-protocol", default=str(DEFAULT_JUDGE_PROTOCOL_PATH))
    audit.add_argument(
        "--judge-annotation-package",
        default=str(DEFAULT_JUDGE_ANNOTATION_PACKAGE_PATH),
    )
    audit.add_argument("--sealed-ops", default=str(DEFAULT_SEALED_OPS_PATH))
    audit.add_argument("--judge-study", default=str(DEFAULT_JUDGE_STUDY_PATH))
    audit.add_argument("--sealed-queue", default=str(DEFAULT_SEALED_QUEUE_PATH))
    audit.add_argument(
        "--external-endpoint-contract",
        default=str(DEFAULT_EXTERNAL_ENDPOINT_CONTRACT_PATH),
    )
    audit.add_argument("--external-systems", default=str(DEFAULT_EXTERNAL_SYSTEMS_PATH))
    audit.add_argument("--claims", default=str(DEFAULT_CLAIMS_MANIFEST_PATH))
    audit.add_argument("--submission-intake", default=str(DEFAULT_SUBMISSION_INTAKE_PATH))
    audit.add_argument("--output", default=None)
    audit.set_defaults(func=cmd_audit)

    readiness = subparsers.add_parser(
        "readiness",
        help="Evaluate release audit data against a readiness profile",
    )
    readiness.add_argument("--scenarios", default=str(DEFAULT_SCENARIO_PATH))
    readiness.add_argument("--audio-manifest", default=str(DEFAULT_AUDIO_MANIFEST_PATH))
    readiness.add_argument("--audio-root", default=".")
    readiness.add_argument("--pricing-manifest", default=str(DEFAULT_PRICING_MANIFEST_PATH))
    readiness.add_argument("--splits", default=str(DEFAULT_SPLIT_MANIFEST_PATH))
    readiness.add_argument("--split-commitments", default=str(DEFAULT_SPLIT_COMMITMENT_PATH))
    readiness.add_argument("--provenance", default=str(DEFAULT_PROVENANCE_MANIFEST_PATH))
    readiness.add_argument("--changelog", default=str(DEFAULT_CHANGELOG_PATH))
    readiness.add_argument("--baseline-manifest", default=str(DEFAULT_BASELINE_MANIFEST_PATH))
    readiness.add_argument("--review-manifest", default=str(DEFAULT_REVIEW_MANIFEST_PATH))
    readiness.add_argument("--judge-protocol", default=str(DEFAULT_JUDGE_PROTOCOL_PATH))
    readiness.add_argument(
        "--judge-annotation-package",
        default=str(DEFAULT_JUDGE_ANNOTATION_PACKAGE_PATH),
    )
    readiness.add_argument("--sealed-ops", default=str(DEFAULT_SEALED_OPS_PATH))
    readiness.add_argument("--judge-study", default=str(DEFAULT_JUDGE_STUDY_PATH))
    readiness.add_argument("--sealed-queue", default=str(DEFAULT_SEALED_QUEUE_PATH))
    readiness.add_argument(
        "--external-endpoint-contract",
        default=str(DEFAULT_EXTERNAL_ENDPOINT_CONTRACT_PATH),
    )
    readiness.add_argument("--external-systems", default=str(DEFAULT_EXTERNAL_SYSTEMS_PATH))
    readiness.add_argument("--claims", default=str(DEFAULT_CLAIMS_MANIFEST_PATH))
    readiness.add_argument("--submission-intake", default=str(DEFAULT_SUBMISSION_INTAKE_PATH))
    readiness.add_argument(
        "--frontier-report",
        default=None,
        help="Optional generated frontier report JSON",
    )
    readiness.add_argument("--run-manifest", default=None, help="Optional frozen run manifest JSON")
    readiness.add_argument(
        "--plot-dir",
        default=None,
        help="Optional frontier plot artifact directory",
    )
    readiness.add_argument(
        "--profile",
        choices=sorted(RELEASE_PROFILES),
        default="seed",
        help="Readiness profile to evaluate",
    )
    readiness.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when the selected profile does not pass",
    )
    readiness.add_argument("--output", default=None)
    readiness.set_defaults(func=cmd_readiness)

    run_manifest = subparsers.add_parser(
        "run-manifest",
        help="Build a frozen run manifest from saved reports",
    )
    run_manifest.add_argument("reports", nargs="+", help="Saved report JSON files")
    run_manifest.add_argument("--audio-manifest", default=str(DEFAULT_AUDIO_MANIFEST_PATH))
    run_manifest.add_argument("--audio-root", default=".")
    run_manifest.add_argument("--pricing-manifest", default=str(DEFAULT_PRICING_MANIFEST_PATH))
    run_manifest.add_argument("--splits", default=str(DEFAULT_SPLIT_MANIFEST_PATH))
    run_manifest.add_argument("--provenance", default=str(DEFAULT_PROVENANCE_MANIFEST_PATH))
    run_manifest.add_argument("--changelog", default=str(DEFAULT_CHANGELOG_PATH))
    run_manifest.add_argument("--baseline-manifest", default=str(DEFAULT_BASELINE_MANIFEST_PATH))
    run_manifest.add_argument("--review-manifest", default=str(DEFAULT_REVIEW_MANIFEST_PATH))
    run_manifest.add_argument("--judge-model", default=None)
    run_manifest.add_argument("--judge-prompt", default=None)
    run_manifest.add_argument("--seed", type=int, default=0)
    run_manifest.add_argument("--region", default=None)
    run_manifest.add_argument("--network", default=None)
    run_manifest.add_argument("--hardware-profile", default=None)
    run_manifest.add_argument("--transport", default=None)
    run_manifest.add_argument("--concurrency-levels", nargs="*", type=int, default=None)
    run_manifest.add_argument("--output", default=None)
    run_manifest.set_defaults(func=cmd_run_manifest)

    return parser


def main() -> None:
    load_workspace_env()
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
