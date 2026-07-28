"""Release-bundle assembly for latency-cost-quality frontier artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any

from src.evaluation.benchmark.changelog import DEFAULT_CHANGELOG_PATH
from src.evaluation.benchmark.frontier import (
    DEFAULT_EXPERIENCE_GATE,
    build_frontier_plot_data,
    build_frontier_report,
    validate_frontier_report,
    validate_frontier_report_file,
    write_frontier_artifacts,
    write_scorecard_artifacts,
)
from src.evaluation.benchmark.openvoicecs import (
    DEFAULT_AUDIO_MANIFEST_PATH,
    DEFAULT_BASELINE_MANIFEST_PATH,
    DEFAULT_REVIEW_MANIFEST_PATH,
    DEFAULT_SCENARIO_PATH,
    build_release_audit,
    load_reports,
)
from src.evaluation.benchmark.pricing import DEFAULT_PRICING_MANIFEST_PATH, load_pricing_manifest
from src.evaluation.benchmark.provenance import DEFAULT_PROVENANCE_MANIFEST_PATH
from src.evaluation.benchmark.readiness import evaluate_release_readiness
from src.evaluation.benchmark.run_manifest import (
    build_run_manifest,
    validate_run_manifest,
    validate_run_manifest_file,
)
from src.evaluation.benchmark.splits import DEFAULT_SPLIT_MANIFEST_PATH


@dataclass(frozen=True)
class ReleaseBundleIssue:
    path: str
    message: str


def build_frontier_release_bundle(
    report_patterns: list[str | Path],
    output_dir: str | Path,
    *,
    scenario_path: str | Path = DEFAULT_SCENARIO_PATH,
    audio_manifest_path: str | Path | None = DEFAULT_AUDIO_MANIFEST_PATH,
    audio_asset_root: str | Path = ".",
    pricing_manifest_path: str | Path | None = DEFAULT_PRICING_MANIFEST_PATH,
    split_manifest_path: str | Path | None = DEFAULT_SPLIT_MANIFEST_PATH,
    provenance_manifest_path: str | Path | None = DEFAULT_PROVENANCE_MANIFEST_PATH,
    changelog_path: str | Path | None = DEFAULT_CHANGELOG_PATH,
    baseline_manifest_path: str | Path | None = DEFAULT_BASELINE_MANIFEST_PATH,
    review_manifest_path: str | Path | None = DEFAULT_REVIEW_MANIFEST_PATH,
    judge_model: str | None = None,
    judge_prompt_path: str | Path | None = None,
    seed: int = 0,
    region: str | None = None,
    network: str | None = None,
    hardware_profile: str | None = None,
    transport: str | None = None,
    concurrency_levels: list[int] | tuple[int, ...] | None = None,
    pricing_snapshot_date: str | None = None,
    experience_gate: float = DEFAULT_EXPERIENCE_GATE,
    utility_weights: dict[str, float] | None = None,
    latency_targets_ms: list[float] | tuple[float, ...] | None = None,
    cost_targets_usd: list[float] | tuple[float, ...] | None = None,
    readiness_profile: str = "seed",
) -> dict[str, Any]:
    """Build a colocated frontier release bundle from saved report JSON files."""
    report_paths = _expand_report_paths(report_patterns)
    reports = load_reports([str(path) for path in report_paths], validate=True)
    if not reports:
        raise ValueError("no reports matched")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plot_dir = output / "plots"
    input_files = _input_file_entries(
        report_paths=report_paths,
        input_dir=output / "inputs",
        scenario_path=scenario_path,
        audio_manifest_path=audio_manifest_path,
        pricing_manifest_path=pricing_manifest_path,
        split_manifest_path=split_manifest_path,
        provenance_manifest_path=provenance_manifest_path,
        changelog_path=changelog_path,
        baseline_manifest_path=baseline_manifest_path,
        review_manifest_path=review_manifest_path,
        judge_prompt_path=judge_prompt_path,
    )

    pricing_manifest = (
        load_pricing_manifest(pricing_manifest_path)
        if pricing_manifest_path is not None
        else None
    )
    environment = {
        "region": region,
        "network": network,
        "hardware_profile": hardware_profile,
        "transport": transport,
        "concurrency_levels": list(concurrency_levels) if concurrency_levels is not None else None,
    }
    frontier = build_frontier_report(
        reports,
        pricing_manifest=pricing_manifest,
        pricing_snapshot_date=pricing_snapshot_date,
        environment=environment,
        experience_gate=experience_gate,
        utility_weights=utility_weights,
        latency_targets_ms=latency_targets_ms,
        cost_targets_usd=cost_targets_usd,
    )
    frontier_path = output / "frontier_report.json"
    _write_json(frontier_path, frontier)
    plot_artifacts = write_frontier_artifacts(frontier, plot_dir)
    scorecard_artifacts = write_scorecard_artifacts(frontier, output / "scorecards")

    run_manifest = build_run_manifest(
        report_paths,
        scenario_path=scenario_path,
        audio_manifest_path=audio_manifest_path,
        audio_asset_root=audio_asset_root,
        pricing_manifest_path=pricing_manifest_path,
        split_manifest_path=split_manifest_path,
        provenance_manifest_path=provenance_manifest_path,
        changelog_path=changelog_path,
        baseline_manifest_path=baseline_manifest_path,
        review_manifest_path=review_manifest_path,
        judge_model=judge_model,
        judge_prompt_path=judge_prompt_path,
        seed=seed,
        region=region,
        network=network,
        hardware_profile=hardware_profile,
        transport=transport,
        concurrency_levels=concurrency_levels,
    )
    _rewrite_run_manifest_to_bundle_inputs(run_manifest, input_files)
    run_manifest_path = output / "run_manifest.json"
    _write_json(run_manifest_path, run_manifest)

    release_audit = build_release_audit(
        scenario_path=scenario_path,
        audio_manifest_path=audio_manifest_path,
        audio_asset_root=audio_asset_root,
        pricing_manifest_path=pricing_manifest_path,
        split_manifest_path=split_manifest_path,
        provenance_manifest_path=provenance_manifest_path,
        changelog_path=changelog_path,
        baseline_manifest_path=baseline_manifest_path,
        review_manifest_path=review_manifest_path,
        sealed_queue_path=None,
        external_systems_path=None,
        claims_manifest_path=None,
        submission_intake_path=None,
    )
    readiness = evaluate_release_readiness(
        release_audit,
        profile=readiness_profile,
        frontier_report=frontier,
        run_manifest=run_manifest,
        run_manifest_base_dir=output,
        verify_run_manifest_files=True,
        plot_dir=plot_dir,
    )
    readiness_path = output / "readiness.json"
    _write_json(readiness_path, readiness)

    frontier_issues = validate_frontier_report(frontier)
    run_manifest_issues = validate_run_manifest(run_manifest)
    bundle = {
        "benchmark": "Latency-Cost-Quality Frontier",
        "bundle_version": "0.1.0",
        "generated_at": time.strftime("%Y-%m-%d"),
        "output_dir": _portable_source_path(output),
        "inputs": {
            "reports": [_portable_source_path(path) for path in report_paths],
            "scenario_path": _portable_source_path(scenario_path),
            "audio_manifest_path": (
                _portable_source_path(audio_manifest_path) if audio_manifest_path else None
            ),
            "pricing_manifest_path": (
                _portable_source_path(pricing_manifest_path) if pricing_manifest_path else None
            ),
            "split_manifest_path": (
                _portable_source_path(split_manifest_path) if split_manifest_path else None
            ),
            "provenance_manifest_path": (
                _portable_source_path(provenance_manifest_path) if provenance_manifest_path else None
            ),
            "changelog_path": _portable_source_path(changelog_path) if changelog_path else None,
            "baseline_manifest_path": (
                _portable_source_path(baseline_manifest_path) if baseline_manifest_path else None
            ),
            "review_manifest_path": (
                _portable_source_path(review_manifest_path) if review_manifest_path else None
            ),
        },
        "input_files": input_files,
        "release_tuple": {
            "seed": seed,
            "judge_model": judge_model,
            "judge_prompt_path": str(judge_prompt_path) if judge_prompt_path else None,
            "environment": environment,
            "readiness_profile": readiness_profile,
        },
        "artifacts": _artifact_entries({
            "frontier_report": frontier_path,
            "run_manifest": run_manifest_path,
            "readiness": readiness_path,
            **{f"plot:{key}": Path(path) for key, path in plot_artifacts.items()},
            **{f"scorecard:{key}": Path(path) for key, path in scorecard_artifacts.items()},
        }, base_dir=output),
        "validation": {
            "passed": (
                not frontier_issues
                and not run_manifest_issues
                and readiness["passed"] is True
            ),
            "frontier_report": _issue_summary(frontier_issues),
            "run_manifest": _issue_summary(run_manifest_issues),
            "readiness": {
                "passed": readiness["passed"],
                "num_issues": readiness["num_issues"],
                "issues": readiness["issues"],
            },
        },
    }
    bundle_path = output / "release_bundle.json"
    _write_json(bundle_path, bundle)
    return bundle


def validate_frontier_release_bundle_file(path: str | Path) -> list[ReleaseBundleIssue]:
    bundle_path = Path(path)
    with open(bundle_path, encoding="utf-8") as f:
        bundle = json.load(f)
    return validate_frontier_release_bundle(bundle, base_dir=bundle_path.parent)


def validate_frontier_release_bundle(
    bundle: dict[str, Any],
    *,
    base_dir: str | Path = ".",
) -> list[ReleaseBundleIssue]:
    """Validate a saved frontier release bundle and its referenced artifacts."""
    issues: list[ReleaseBundleIssue] = []
    if not isinstance(bundle, dict):
        return [ReleaseBundleIssue("<root>", "must be an object")]
    for field in (
        "benchmark",
        "bundle_version",
        "generated_at",
        "inputs",
        "release_tuple",
        "input_files",
        "artifacts",
        "validation",
    ):
        if field not in bundle:
            issues.append(ReleaseBundleIssue(field, "missing required field"))
    if bundle.get("benchmark") != "Latency-Cost-Quality Frontier":
        issues.append(ReleaseBundleIssue("benchmark", "must be Latency-Cost-Quality Frontier"))

    _validate_input_files(issues, bundle.get("input_files"), base_dir=Path(base_dir))

    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        issues.append(ReleaseBundleIssue("artifacts", "must be a non-empty object"))
        return issues

    for required in (
        "frontier_report",
        "run_manifest",
        "readiness",
        "plot:plot_data",
        "scorecard:json",
        "scorecard:csv",
        "scorecard:markdown",
    ):
        if required not in artifacts:
            issues.append(ReleaseBundleIssue(f"artifacts.{required}", "missing required artifact"))

    resolved: dict[str, Path] = {}
    for name, entry in artifacts.items():
        path = _validate_artifact_entry(
            issues,
            f"artifacts.{name}",
            entry,
            base_dir=Path(base_dir),
        )
        if path is not None:
            resolved[name] = path

    frontier_report = None
    frontier_path = resolved.get("frontier_report")
    if frontier_path is not None:
        loaded_frontier = _load_json_artifact(
            issues,
            "artifacts.frontier_report",
            frontier_path,
        )
        if isinstance(loaded_frontier, dict):
            frontier_report = loaded_frontier
        for issue in validate_frontier_report_file(frontier_path):
            issues.append(
                ReleaseBundleIssue(
                    f"artifacts.frontier_report.{issue.path}",
                    issue.message,
                )
            )

    run_manifest_path = resolved.get("run_manifest")
    if run_manifest_path is not None:
        for issue in validate_run_manifest_file(run_manifest_path):
            issues.append(ReleaseBundleIssue(f"artifacts.run_manifest.{issue.path}", issue.message))

    readiness_path = resolved.get("readiness")
    if readiness_path is not None:
        readiness = _load_json_artifact(issues, "artifacts.readiness", readiness_path)
        if isinstance(readiness, dict):
            _validate_readiness_artifact(issues, readiness)

    plot_data_path = resolved.get("plot:plot_data")
    if plot_data_path is not None:
        plot_data = _load_json_artifact(issues, "artifacts.plot:plot_data", plot_data_path)
        if isinstance(plot_data, dict) and not isinstance(plot_data.get("domains"), dict):
            issues.append(
                ReleaseBundleIssue(
                    "artifacts.plot:plot_data.domains",
                    "must be an object",
                )
            )
        elif isinstance(plot_data, dict) and isinstance(frontier_report, dict):
            expected_plot_data = build_frontier_plot_data(frontier_report)
            if plot_data != expected_plot_data:
                issues.append(
                    ReleaseBundleIssue(
                        "artifacts.plot:plot_data",
                        "must match plot data regenerated from frontier_report",
                    )
                )

    scorecard_json_path = resolved.get("scorecard:json")
    if scorecard_json_path is not None:
        rows = _load_json_list_artifact(issues, "artifacts.scorecard:json", scorecard_json_path)
        if isinstance(rows, list) and not rows:
            issues.append(
                ReleaseBundleIssue(
                    "artifacts.scorecard:json",
                    "must contain scorecard rows",
                )
            )
        elif isinstance(rows, list):
            for index, row in enumerate(rows):
                if not isinstance(row, dict) or not row.get("system"):
                    issues.append(
                        ReleaseBundleIssue(
                            f"artifacts.scorecard:json[{index}]",
                            "must be an object with a system name",
                        )
                    )

    scorecard_csv_path = resolved.get("scorecard:csv")
    if scorecard_csv_path is not None:
        _validate_text_prefix(
            issues,
            "artifacts.scorecard:csv",
            scorecard_csv_path,
            expected_prefix="system,frontier_eligible,",
        )

    scorecard_markdown_path = resolved.get("scorecard:markdown")
    if scorecard_markdown_path is not None:
        _validate_text_prefix(
            issues,
            "artifacts.scorecard:markdown",
            scorecard_markdown_path,
            expected_prefix="# Latency-Cost-Quality Scorecards",
        )

    for name, path in resolved.items():
        if name.startswith("plot:") and name != "plot:plot_data":
            try:
                prefix = path.read_text(encoding="utf-8", errors="ignore").lstrip()[:4]
            except OSError:
                continue
            if prefix != "<svg":
                issues.append(ReleaseBundleIssue(f"artifacts.{name}", "must be an SVG file"))

    validation = bundle.get("validation")
    if not isinstance(validation, dict):
        issues.append(ReleaseBundleIssue("validation", "must be an object"))
    elif "passed" not in validation:
        issues.append(ReleaseBundleIssue("validation.passed", "missing required field"))
    elif not isinstance(validation.get("passed"), bool):
        issues.append(ReleaseBundleIssue("validation.passed", "must be boolean"))

    return issues


def _expand_report_paths(patterns: list[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob(str(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            candidate = Path(pattern)
            if candidate.exists():
                paths.append(candidate)
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise ValueError("no reports matched")
    return unique


def _input_file_entries(
    *,
    report_paths: list[Path],
    input_dir: Path,
    scenario_path: str | Path,
    audio_manifest_path: str | Path | None,
    pricing_manifest_path: str | Path | None,
    split_manifest_path: str | Path | None,
    provenance_manifest_path: str | Path | None,
    changelog_path: str | Path | None,
    baseline_manifest_path: str | Path | None,
    review_manifest_path: str | Path | None,
    judge_prompt_path: str | Path | None,
) -> dict[str, Any]:
    report_dir = input_dir / "reports"
    manifest_dir = input_dir / "manifests"
    base_dir = input_dir.parent
    return {
        "reports": [
            _snapshot_artifact_entry(
                path,
                report_dir / f"{index:03d}_{path.name}",
                base_dir=base_dir,
            )
            for index, path in enumerate(report_paths, 1)
        ],
        "scenario": _snapshot_artifact_entry(
            Path(scenario_path),
            manifest_dir / "scenarios.json",
            base_dir=base_dir,
        ),
        "audio_manifest": _optional_snapshot_artifact_entry(
            audio_manifest_path,
            manifest_dir / "audio_manifest.json",
            base_dir=base_dir,
        ),
        "pricing_manifest": _optional_snapshot_artifact_entry(
            pricing_manifest_path,
            manifest_dir / "pricing_manifest.json",
            base_dir=base_dir,
        ),
        "split_manifest": _optional_snapshot_artifact_entry(
            split_manifest_path,
            manifest_dir / "splits.json",
            base_dir=base_dir,
        ),
        "provenance_manifest": _optional_snapshot_artifact_entry(
            provenance_manifest_path,
            manifest_dir / "provenance.json",
            base_dir=base_dir,
        ),
        "changelog": _optional_snapshot_artifact_entry(
            changelog_path,
            manifest_dir / "changelog.json",
            base_dir=base_dir,
        ),
        "baseline_manifest": _optional_snapshot_artifact_entry(
            baseline_manifest_path,
            manifest_dir / "reference_baselines.json",
            base_dir=base_dir,
        ),
        "review_manifest": _optional_snapshot_artifact_entry(
            review_manifest_path,
            manifest_dir / "scenario_reviews.json",
            base_dir=base_dir,
        ),
        "judge_prompt": _optional_snapshot_artifact_entry(
            judge_prompt_path,
            manifest_dir / "judge_prompt.txt",
            base_dir=base_dir,
        ),
    }


def _rewrite_run_manifest_to_bundle_inputs(
    run_manifest: dict[str, Any],
    input_files: dict[str, Any],
) -> None:
    """Point a bundle's run manifest at the snapshotted input files."""
    reports = run_manifest.get("reports")
    snapshotted_reports = input_files.get("reports")
    if isinstance(reports, list) and isinstance(snapshotted_reports, list):
        for report, snapshotted_report in zip(reports, snapshotted_reports, strict=False):
            _copy_entry_path(report, snapshotted_report)

    release_tuple = run_manifest.get("release_tuple")
    if not isinstance(release_tuple, dict):
        return

    for manifest_field, input_field in (
        ("scenario_suite", "scenario"),
        ("audio_manifest", "audio_manifest"),
        ("pricing_manifest", "pricing_manifest"),
        ("split_manifest", "split_manifest"),
        ("provenance_manifest", "provenance_manifest"),
        ("changelog", "changelog"),
        ("baseline_manifest", "baseline_manifest"),
        ("review_manifest", "review_manifest"),
    ):
        _copy_entry_path(release_tuple.get(manifest_field), input_files.get(input_field))

    judge = release_tuple.get("judge")
    if isinstance(judge, dict):
        _copy_entry_path(judge.get("prompt"), input_files.get("judge_prompt"))


def _copy_entry_path(target: Any, source: Any) -> None:
    if not isinstance(target, dict) or not isinstance(source, dict):
        return
    source_path = source.get("path")
    if isinstance(source_path, str) and source_path:
        target["path"] = source_path


def _optional_snapshot_artifact_entry(
    path: str | Path | None,
    destination: Path,
    *,
    base_dir: Path,
) -> dict[str, Any] | None:
    if path is None:
        return None
    return _snapshot_artifact_entry(Path(path), destination, base_dir=base_dir)


def _snapshot_artifact_entry(source: Path, destination: Path, *, base_dir: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    entry = _artifact_entry(destination, base_dir=base_dir)
    entry["source_path"] = _portable_source_path(source)
    return entry


def _portable_source_path(source: Path) -> str:
    """Record where an input came from without leaking the builder's filesystem.

    ``source_path`` is audit context, not a locator — the bundle validates against
    its own ``inputs/`` snapshot. An absolute path here publishes the operator's
    home directory and username into a public release artifact, which has already
    had to be scrubbed once (see the repository history). Relative to the current
    working directory keeps the provenance and drops the machine.
    """
    resolved = Path(source).resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        # Outside the working tree: keep only the final two components so the
        # entry stays meaningful without naming intermediate directories.
        return str(Path(*resolved.parts[-2:]))


def _validate_input_files(
    issues: list[ReleaseBundleIssue],
    input_files: Any,
    *,
    base_dir: Path,
) -> None:
    if not isinstance(input_files, dict):
        issues.append(ReleaseBundleIssue("input_files", "must be an object"))
        return
    reports = input_files.get("reports")
    if not isinstance(reports, list) or not reports:
        issues.append(ReleaseBundleIssue("input_files.reports", "must be a non-empty list"))
    elif len(reports) != len({entry.get("path") for entry in reports if isinstance(entry, dict)}):
        issues.append(ReleaseBundleIssue("input_files.reports", "must not contain duplicate paths"))
    if isinstance(reports, list):
        for index, entry in enumerate(reports):
            _validate_artifact_entry(
                issues,
                f"input_files.reports[{index}]",
                entry,
                base_dir=base_dir,
            )

    required_inputs = ("scenario",)
    optional_inputs = (
        "audio_manifest",
        "pricing_manifest",
        "split_manifest",
        "provenance_manifest",
        "changelog",
        "baseline_manifest",
        "review_manifest",
        "judge_prompt",
    )
    for name in required_inputs:
        if name not in input_files:
            issues.append(ReleaseBundleIssue(f"input_files.{name}", "missing required input"))
            continue
        _validate_artifact_entry(
            issues,
            f"input_files.{name}",
            input_files[name],
            base_dir=base_dir,
        )
    for name in optional_inputs:
        if name not in input_files or input_files[name] is None:
            continue
        _validate_artifact_entry(
            issues,
            f"input_files.{name}",
            input_files[name],
            base_dir=base_dir,
        )


def _artifact_entries(paths: dict[str, Path], *, base_dir: Path) -> dict[str, dict[str, Any]]:
    return {name: _artifact_entry(path, base_dir=base_dir) for name, path in sorted(paths.items())}


def _artifact_entry(path: Path, *, base_dir: Path | None = None) -> dict[str, Any]:
    stored_path = str(path)
    if base_dir is not None:
        try:
            stored_path = str(path.relative_to(base_dir))
        except ValueError:
            stored_path = str(path)
    return {
        "path": stored_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def _validate_artifact_entry(
    issues: list[ReleaseBundleIssue],
    path: str,
    entry: Any,
    *,
    base_dir: Path,
) -> Path | None:
    if not isinstance(entry, dict):
        issues.append(ReleaseBundleIssue(path, "must be an object"))
        return None
    for field in ("path", "sha256", "bytes"):
        if field not in entry:
            issues.append(ReleaseBundleIssue(f"{path}.{field}", "missing required field"))
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        issues.append(ReleaseBundleIssue(f"{path}.path", "must be a non-empty string"))
        return None
    artifact_path = _resolve_artifact_path(raw_path, base_dir)
    if not artifact_path.exists():
        issues.append(ReleaseBundleIssue(f"{path}.path", "file does not exist"))
        return None
    expected_sha = entry.get("sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        issues.append(ReleaseBundleIssue(f"{path}.sha256", "must be a SHA-256 hex digest"))
    else:
        actual_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            issues.append(ReleaseBundleIssue(f"{path}.sha256", "does not match file contents"))
    expected_bytes = entry.get("bytes")
    if not isinstance(expected_bytes, int) or expected_bytes < 0:
        issues.append(ReleaseBundleIssue(f"{path}.bytes", "must be a nonnegative integer"))
    elif artifact_path.stat().st_size != expected_bytes:
        issues.append(ReleaseBundleIssue(f"{path}.bytes", "does not match file size"))
    return artifact_path


def _resolve_artifact_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return path


def _load_json_artifact(
    issues: list[ReleaseBundleIssue],
    path: str,
    artifact_path: Path,
) -> dict[str, Any] | None:
    try:
        with open(artifact_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        issues.append(ReleaseBundleIssue(path, "must be valid JSON"))
        return None
    if not isinstance(data, dict):
        issues.append(ReleaseBundleIssue(path, "must be an object"))
        return None
    return data


def _load_json_list_artifact(
    issues: list[ReleaseBundleIssue],
    path: str,
    artifact_path: Path,
) -> list[Any] | None:
    try:
        with open(artifact_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        issues.append(ReleaseBundleIssue(path, "must be valid JSON"))
        return None
    if not isinstance(data, list):
        issues.append(ReleaseBundleIssue(path, "must be a list"))
        return None
    return data


def _validate_text_prefix(
    issues: list[ReleaseBundleIssue],
    path: str,
    artifact_path: Path,
    *,
    expected_prefix: str,
) -> None:
    try:
        prefix = artifact_path.read_text(encoding="utf-8", errors="ignore")[:len(expected_prefix)]
    except OSError:
        return
    if prefix != expected_prefix:
        issues.append(ReleaseBundleIssue(path, f"must start with {expected_prefix!r}"))


def _validate_readiness_artifact(
    issues: list[ReleaseBundleIssue],
    readiness: dict[str, Any],
) -> None:
    for field in ("profile", "passed", "num_issues", "issues", "artifacts"):
        if field not in readiness:
            issues.append(
                ReleaseBundleIssue(
                    f"artifacts.readiness.{field}",
                    "missing required field",
                )
            )
    if "passed" in readiness and not isinstance(readiness.get("passed"), bool):
        issues.append(ReleaseBundleIssue("artifacts.readiness.passed", "must be boolean"))
    if "num_issues" in readiness and (
        not isinstance(readiness.get("num_issues"), int)
        or readiness["num_issues"] < 0
    ):
        issues.append(
            ReleaseBundleIssue(
                "artifacts.readiness.num_issues",
                "must be nonnegative integer",
            )
        )
    if "issues" in readiness and not isinstance(readiness.get("issues"), list):
        issues.append(ReleaseBundleIssue("artifacts.readiness.issues", "must be a list"))


def _issue_summary(issues: list[Any]) -> dict[str, Any]:
    return {
        "passed": not issues,
        "num_issues": len(issues),
        "issues": [
            {
                "path": issue.path,
                "message": issue.message,
            }
            for issue in issues
        ],
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
