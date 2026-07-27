"""Reference baseline report packages for OpenVoiceCS-Bench."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.benchmark.openvoicecs import (
    DEFAULT_AUDIO_MANIFEST_PATH,
    DEFAULT_SCENARIO_PATH,
    OpenVoiceCSBench,
    no_op_agent,
    oracle_agent,
    validate_report,
)

DEFAULT_BASELINE_DIR = Path("data/openvoicecs/baselines")
DEFAULT_BASELINE_MANIFEST_PATH = DEFAULT_BASELINE_DIR / "reference_baselines_v0.1.json"


@dataclass(frozen=True)
class BaselineIssue:
    """Structured reference-baseline validation issue."""

    item_id: str
    path: str
    message: str


BASELINE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "oracle_text",
        "agent": "oracle",
        "mode": "text",
        "description": "Reference ceiling over all text scenarios.",
    },
    {
        "id": "noop_text",
        "agent": "noop",
        "mode": "text",
        "description": "Polite non-action baseline over all text scenarios.",
    },
    {
        "id": "oracle_audio_manifest",
        "agent": "oracle",
        "mode": "audio_manifest",
        "description": "Reference ceiling over all audio manifest variants.",
    },
    {
        "id": "noop_audio_manifest",
        "agent": "noop",
        "mode": "audio_manifest",
        "description": "Polite non-action baseline over all audio manifest variants.",
    },
)


def build_reference_baselines(
    *,
    scenario_path: str | Path = DEFAULT_SCENARIO_PATH,
    audio_manifest_path: str | Path | None = DEFAULT_AUDIO_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_BASELINE_DIR,
    manifest_path: str | Path = DEFAULT_BASELINE_MANIFEST_PATH,
    trials: int = 3,
) -> dict[str, Any]:
    """Generate deterministic reference baseline reports and a manifest."""
    if trials < 1:
        raise ValueError("trials must be >= 1")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bench = OpenVoiceCSBench.load(scenario_path)

    baseline_entries = []
    for spec in BASELINE_SPECS:
        if spec["mode"] == "audio_manifest" and audio_manifest_path is None:
            continue
        report = _run_baseline_report(
            bench,
            spec,
            audio_manifest_path=audio_manifest_path,
            trials=trials,
        )
        report_path = output / f"{spec['id']}.json"
        _write_json(report_path, report)
        baseline_entries.append(
            {
                "id": spec["id"],
                "agent": spec["agent"],
                "mode": spec["mode"],
                "description": spec["description"],
                "trials": trials,
                "report": _file_entry(report_path),
                "expected": _baseline_summary(report),
            }
        )

    manifest = {
        "name": "OpenVoiceCS-Bench Reference Baselines",
        "version": "0.1.0",
        "benchmark_version": bench.version,
        "generated_at": time.strftime("%Y-%m-%d"),
        "scenario_file": _file_entry(scenario_path),
        "audio_manifest_file": (
            _file_entry(audio_manifest_path) if audio_manifest_path is not None else None
        ),
        "baselines": baseline_entries,
    }
    _write_json(manifest_path, manifest)
    return manifest


def validate_reference_baselines_file(
    path: str | Path = DEFAULT_BASELINE_MANIFEST_PATH,
) -> list[BaselineIssue]:
    """Validate a saved baseline manifest and referenced reports."""
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    return validate_reference_baselines(manifest)


def validate_reference_baselines(manifest: dict[str, Any]) -> list[BaselineIssue]:
    """Validate a reference baseline manifest and report hashes."""
    issues: list[BaselineIssue] = []
    if not isinstance(manifest, dict):
        return [BaselineIssue("<baselines>", "<root>", "must be an object")]
    for field in ("name", "version", "benchmark_version", "scenario_file", "baselines"):
        if field not in manifest:
            issues.append(BaselineIssue("<baselines>", field, "missing required field"))
    if issues:
        return issues
    if manifest.get("name") != "OpenVoiceCS-Bench Reference Baselines":
        issues.append(BaselineIssue("<baselines>", "name", "unsupported baseline manifest"))
    _validate_file_entry(issues, "<baselines>", "scenario_file", manifest.get("scenario_file"))
    if manifest.get("audio_manifest_file") is not None:
        _validate_file_entry(
            issues,
            "<baselines>",
            "audio_manifest_file",
            manifest.get("audio_manifest_file"),
        )

    baselines = manifest.get("baselines")
    if not isinstance(baselines, list) or not baselines:
        issues.append(BaselineIssue("<baselines>", "baselines", "must be a non-empty list"))
        return issues
    seen_ids: set[str] = set()
    present_ids = set()
    for index, baseline in enumerate(baselines):
        _validate_baseline_entry(issues, baseline, index=index, seen_ids=seen_ids)
        if isinstance(baseline, dict) and isinstance(baseline.get("id"), str):
            present_ids.add(baseline["id"])
    required_ids = {spec["id"] for spec in BASELINE_SPECS}
    missing = required_ids - present_ids
    for baseline_id in sorted(missing):
        issues.append(BaselineIssue(baseline_id, "baselines", "missing required baseline"))
    return issues


def reference_baseline_stats(manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize reference baseline manifest scores for release metadata."""
    if not isinstance(manifest, dict):
        return {
            "present": False,
            "num_baselines": 0,
        }
    baselines = manifest.get("baselines", [])
    baselines = baselines if isinstance(baselines, list) else []
    return {
        "present": True,
        "version": manifest.get("version"),
        "benchmark_version": manifest.get("benchmark_version"),
        "num_baselines": len(baselines),
        "baselines": {
            baseline.get("id", f"baseline_{index}"): baseline.get("expected", {})
            for index, baseline in enumerate(baselines)
            if isinstance(baseline, dict)
        },
    }


def _run_baseline_report(
    bench: OpenVoiceCSBench,
    spec: dict[str, Any],
    *,
    audio_manifest_path: str | Path | None,
    trials: int,
) -> dict[str, Any]:
    agent_fn = _metered_agent(_agent_function(spec["agent"]))
    metadata = {
        "display_name": spec["id"],
        "provider": "openvoicecs_reference",
        "model_id": f"{spec['id']}-v0.1",
        "agent": spec["agent"],
        "baseline_id": spec["id"],
        "pipeline_type": "cascaded",
        "input_modality": "audio" if spec["mode"] == "audio_manifest" else "text",
        "pricing_profile_id": "reference-zero-v0.1",
        "pricing_snapshot_date": "2026-06-11",
    }
    if spec["mode"] == "audio_manifest":
        report = bench.score_audio_manifest(
            agent_fn,
            manifest_path=audio_manifest_path,
            trials=trials,
            model_metadata=metadata,
        )
    else:
        report = bench.score_agent(
            agent_fn,
            trials=trials,
            model_metadata=metadata,
        )
    report["elapsed_seconds"] = 0.0
    report["baseline_id"] = spec["id"]
    return report


def _validate_baseline_entry(
    issues: list[BaselineIssue],
    baseline: Any,
    *,
    index: int,
    seen_ids: set[str],
) -> None:
    path = f"baselines[{index}]"
    if not isinstance(baseline, dict):
        issues.append(BaselineIssue(path, path, "must be an object"))
        return
    baseline_id = baseline.get("id")
    if not isinstance(baseline_id, str) or not baseline_id.strip():
        issues.append(BaselineIssue(path, f"{path}.id", "must be a non-empty string"))
        baseline_id = path
    elif baseline_id in seen_ids:
        issues.append(BaselineIssue(baseline_id, f"{path}.id", "duplicate baseline id"))
    seen_ids.add(str(baseline_id))

    if baseline.get("agent") not in {"oracle", "noop"}:
        issues.append(BaselineIssue(str(baseline_id), f"{path}.agent", "must be oracle or noop"))
    if baseline.get("mode") not in {"text", "audio_manifest"}:
        issues.append(
            BaselineIssue(
                str(baseline_id),
                f"{path}.mode",
                "must be text or audio_manifest",
            )
        )
    if not isinstance(baseline.get("trials"), int) or baseline.get("trials", 0) < 1:
        issues.append(
            BaselineIssue(
                str(baseline_id),
                f"{path}.trials",
                "must be a positive integer",
            )
        )

    report_entry = baseline.get("report")
    _validate_file_entry(issues, str(baseline_id), f"{path}.report", report_entry)
    report = _load_report_from_entry(report_entry)
    if report is None:
        return
    for report_issue in validate_report(report):
        issues.append(
            BaselineIssue(
                str(baseline_id),
                f"{path}.report.{report_issue.path}",
                report_issue.message,
            )
        )
    expected = baseline.get("expected")
    if not isinstance(expected, dict):
        issues.append(BaselineIssue(str(baseline_id), f"{path}.expected", "must be an object"))
        return
    actual = _baseline_summary(report)
    for field, actual_value in actual.items():
        if expected.get(field) != actual_value:
            issues.append(
                BaselineIssue(
                    str(baseline_id),
                    f"{path}.expected.{field}",
                    "does not match report",
                )
            )
    _validate_baseline_expectations(issues, str(baseline_id), baseline, report, path=path)


def _validate_baseline_expectations(
    issues: list[BaselineIssue],
    baseline_id: str,
    baseline: dict[str, Any],
    report: dict[str, Any],
    *,
    path: str,
) -> None:
    metadata = report.get("model_metadata", {})
    if isinstance(metadata, dict) and metadata.get("baseline_id") != baseline.get("id"):
        issues.append(
            BaselineIssue(
                baseline_id,
                f"{path}.report.model_metadata.baseline_id",
                "does not match baseline id",
            )
        )
    if baseline.get("agent") == "oracle":
        if report.get("overall_score") != 100.0:
            issues.append(
                BaselineIssue(
                    baseline_id,
                    f"{path}.report.overall_score",
                    "oracle baseline must score 100",
                )
            )
        if report.get("pass_k") != 1.0:
            issues.append(
                BaselineIssue(
                    baseline_id,
                    f"{path}.report.pass_k",
                    "oracle baseline must pass every trial",
                )
            )
    if baseline.get("agent") == "noop" and report.get("pass_at_k") != 0.0:
        issues.append(
            BaselineIssue(
                baseline_id,
                f"{path}.report.pass_at_k",
                "noop baseline must not pass",
            )
        )
    if (
        baseline.get("mode") == "audio_manifest"
        and report.get("evaluation_mode") != "audio_manifest"
    ):
        issues.append(
            BaselineIssue(
                baseline_id,
                f"{path}.report.evaluation_mode",
                "must be audio_manifest",
            )
        )


def _baseline_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall_score": report.get("overall_score"),
        "pass_at_k": report.get("pass_at_k"),
        "pass_k": report.get("pass_k"),
        "mean_pass_rate": report.get("mean_pass_rate"),
        "num_scenarios": report.get("num_scenarios"),
        "num_trials_per_scenario": report.get("num_trials_per_scenario"),
        "task_success": (report.get("metric_scores") or {}).get("task_success"),
        "safety": (report.get("metric_scores") or {}).get("safety"),
    }


def _load_report_from_entry(entry: Any) -> dict[str, Any] | None:
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        return None
    path = Path(entry["path"])
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    return report if isinstance(report, dict) else None


def _validate_file_entry(
    issues: list[BaselineIssue],
    item_id: str,
    path: str,
    entry: Any,
) -> None:
    if not isinstance(entry, dict):
        issues.append(BaselineIssue(item_id, path, "must be an object"))
        return
    for field in ("path", "sha256", "bytes"):
        if field not in entry:
            issues.append(BaselineIssue(item_id, f"{path}.{field}", "missing required field"))
    if "sha256" in entry and not _looks_sha256(entry.get("sha256")):
        issues.append(BaselineIssue(item_id, f"{path}.sha256", "must be a SHA-256 hex digest"))
    if "bytes" in entry and (
        isinstance(entry.get("bytes"), bool)
        or not isinstance(entry.get("bytes"), int)
        or entry["bytes"] < 0
    ):
        issues.append(BaselineIssue(item_id, f"{path}.bytes", "must be a nonnegative integer"))
    if isinstance(entry.get("path"), str):
        _validate_file_contents(issues, item_id, path, entry)


def _validate_file_contents(
    issues: list[BaselineIssue],
    item_id: str,
    path: str,
    entry: dict[str, Any],
) -> None:
    file_path = Path(entry["path"])
    if not file_path.exists():
        issues.append(BaselineIssue(item_id, f"{path}.path", "file does not exist"))
        return
    data = file_path.read_bytes()
    if _looks_sha256(entry.get("sha256")) and hashlib.sha256(data).hexdigest() != entry["sha256"]:
        issues.append(BaselineIssue(item_id, f"{path}.sha256", "does not match file contents"))
    if isinstance(entry.get("bytes"), int) and len(data) != entry["bytes"]:
        issues.append(BaselineIssue(item_id, f"{path}.bytes", "does not match file size"))


def _file_entry(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _agent_function(agent: str) -> Callable[[dict[str, Any], int], Any]:
    if agent == "oracle":
        return oracle_agent
    if agent == "noop":
        return no_op_agent
    raise ValueError(f"unsupported baseline agent: {agent}")


def _metered_agent(
    agent_fn: Callable[[dict[str, Any], int], Any],
) -> Callable[[dict[str, Any], int], Any]:
    """Attach deterministic zero-cost telemetry to reference baseline traces."""

    def wrapped(scenario: dict[str, Any], trial_index: int) -> dict[str, Any]:
        trace = dict(agent_fn(scenario, trial_index))
        text = " ".join(
            str(message.get("text", ""))
            for message in trace.get("messages", [])
            if isinstance(message, dict)
        )
        latency_ms = trace.get("latency_ms") or scenario.get("experience", {}).get(
            "reference_latency_ms",
            750,
        )
        duration_seconds = max(float(latency_ms) / 1000.0, 0.001)
        trace.setdefault("cost_usd", 0.0)
        trace.setdefault(
            "usage",
            {
                "input_tokens": max(1, len(str(scenario.get("customer_goal", "")).split())),
                "output_tokens": max(1, len(text.split())),
                "tts_characters": max(1, len(text)),
                "asr_seconds": round(duration_seconds, 3),
                "tts_seconds": round(duration_seconds, 3),
                "telephony_seconds": round(duration_seconds, 3),
                "transport_seconds": round(duration_seconds, 3),
            },
        )
        return trace

    return wrapped


def _looks_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )
