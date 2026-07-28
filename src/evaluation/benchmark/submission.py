"""External submission adapter helpers for OpenVoiceCS-Bench.

Submissions are plain Python callables loaded from ``path.py:function_name``.
The callable receives ``(scenario, trial_index)`` and returns the same trace
shape used by built-in agents.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import textwrap
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.benchmark.datapaths import data_path
from src.evaluation.benchmark.openvoicecs import (
    DEFAULT_AUDIO_MANIFEST_PATH,
    DEFAULT_SCENARIO_PATH,
    OpenVoiceCSBench,
)
from src.evaluation.benchmark.provider_adapters import (
    ProviderSpec,
    build_provider_agent,
    provider_metadata,
)

SubmissionFn = Callable[[dict[str, Any], int], Any]
PIPELINE_TYPES = {"cascaded", "native_speech_to_speech", "unknown"}
INPUT_MODALITIES = {"text", "audio", "multimodal", "unknown"}
DEFAULT_SUBMISSION_INTAKE_PATH = data_path("submissions", "reference_submission_intake_v0.1.json")
SUBMISSION_STATUSES = {"reference_fixture", "pending_review", "official", "rejected", "retired"}
SUBMISSION_REQUIRED_ARTIFACTS = (
    "submission_card",
    "report",
    "run_manifest",
    "release_bundle",
    "external_systems_registry",
    "judge_annotation_package",
    "leaderboard_claims",
)


@dataclass(frozen=True)
class SubmissionCardIssue:
    """Structured submission-card validation issue."""

    item_id: str
    path: str
    message: str


@dataclass(frozen=True)
class SubmissionIntakeIssue:
    """Structured submission-intake envelope validation issue."""

    item_id: str
    path: str
    message: str


def build_submission_template(*, function_name: str = "run") -> str:
    """Return a starter OpenVoiceCS submission adapter module."""
    if not function_name.isidentifier():
        raise ValueError("function_name must be a valid Python identifier")
    return textwrap.dedent(f'''\
        """Starter OpenVoiceCS-Bench submission adapter.

        Score locally with:
            python scripts/run_openvoicecs.py submit path/to/adapter.py:{function_name} \\
              --name my_submission --trials 1
        """

        from __future__ import annotations


        def {function_name}(scenario: dict, trial_index: int) -> dict:
            """Return one agent trace for a scenario or audio variant.

            Replace this stub with your production voice-agent call. For text
            scenarios, `scenario["user_utterance"]` contains the user request.
            For audio-mode runs, `scenario["audio_variant"]` contains manifest
            metadata such as transcript, file path, and perturbations.
            """
            del trial_index
            user_text = scenario.get("user_utterance")
            audio_variant = scenario.get("audio_variant") or {{}}
            if not user_text:
                user_text = audio_variant.get("transcript", "")

            # TODO: call your ASR/LLM/TTS or native speech-to-speech stack here.
            response_text = "I can help with that."

            return {{
                "messages": [
                    {{"role": "user", "text": user_text}},
                    {{"role": "agent", "text": response_text}},
                ],
                "tool_calls": [],
                "events": [],
                "latency": {{
                    "v2v_ttfb_ms": None,
                    "v2v_last_byte_ms": None,
                    "barge_in_stop_ms": None,
                    "interruption_recovery_ms": None,
                    "stage_latency_ms": {{
                        "asr_finalization_ms": None,
                        "llm_ttft_ms": None,
                        "tts_first_chunk_ms": None,
                    }},
                }},
                "usage": {{
                    "asr_seconds": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "tts_characters": None,
                    "call_duration_seconds": None,
                    "transport_seconds": None,
                }},
                "cost_usd": None,
            }}
        ''')


def write_submission_template(
    path: str | Path,
    *,
    function_name: str = "run",
    overwrite: bool = False,
) -> Path:
    """Write a starter submission adapter and return the resolved path."""
    output = Path(path).expanduser()
    if output.suffix != ".py":
        raise ValueError("submission template path must end in .py")
    if output.exists() and not overwrite:
        raise FileExistsError(f"submission template already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        build_submission_template(function_name=function_name),
        encoding="utf-8",
    )
    return output.resolve()


def load_submission_callable(spec: str) -> SubmissionFn:
    """Load a submission callable from ``/path/to/file.py:function``."""
    if ":" not in spec:
        raise ValueError("submission spec must have the form path.py:function")
    path_raw, function_name = spec.rsplit(":", 1)
    if not path_raw or not function_name:
        raise ValueError("submission spec must include both path and function")

    path = Path(path_raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"submission file not found: {path}")
    if path.suffix != ".py":
        raise ValueError("submission file must be a .py file")

    module_name = f"openvoicecs_submission_{abs(hash(str(path)))}"
    spec_obj = importlib.util.spec_from_file_location(module_name, path)
    if spec_obj is None or spec_obj.loader is None:
        raise ImportError(f"could not import submission module: {path}")

    module = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(module)
    fn = getattr(module, function_name, None)
    if not callable(fn):
        raise AttributeError(f"submission callable not found: {function_name}")
    return fn


def score_submission(
    submission_spec: str,
    *,
    scenario_path: str | Path = DEFAULT_SCENARIO_PATH,
    audio_manifest_path: str | Path = DEFAULT_AUDIO_MANIFEST_PATH,
    mode: str = "text",
    max_items: int | None = None,
    trials: int = 1,
    track: str | None = None,
    submission_name: str | None = None,
    provider: str | None = None,
    model_id: str | None = None,
    pricing_profile_id: str | None = None,
    pricing_snapshot_date: str | None = None,
    pipeline_type: str | None = None,
) -> dict[str, Any]:
    """Score an external submission against text scenarios or audio variants."""
    if mode not in {"text", "audio"}:
        raise ValueError("mode must be 'text' or 'audio'")
    if pipeline_type is not None and pipeline_type not in PIPELINE_TYPES:
        raise ValueError(f"pipeline_type must be one of: {', '.join(sorted(PIPELINE_TYPES))}")
    fn = load_submission_callable(submission_spec)
    bench = OpenVoiceCSBench.load(scenario_path)
    metadata = {
        "agent": submission_name or submission_spec,
        "submission_spec": submission_spec,
        "submission_mode": mode,
        "provider": provider,
        "model_id": model_id,
        "pricing_profile_id": pricing_profile_id,
        "pricing_snapshot_date": pricing_snapshot_date,
        "pipeline_type": pipeline_type or "unknown",
    }
    if mode == "audio":
        metadata["input_modality"] = "audio"
        return bench.score_audio_manifest(
            fn,
            manifest_path=audio_manifest_path,
            max_variants=max_items,
            trials=trials,
            track=track,
            model_metadata=metadata,
        )
    return bench.score_agent(
        fn,
        max_scenarios=max_items,
        trials=trials,
        track=track,
        model_metadata=metadata,
    )


def score_provider(
    provider_spec: ProviderSpec,
    *,
    scenario_path: str | Path = DEFAULT_SCENARIO_PATH,
    audio_manifest_path: str | Path = DEFAULT_AUDIO_MANIFEST_PATH,
    mode: str = "text",
    max_items: int | None = None,
    trials: int = 1,
    track: str | None = None,
    pricing_profile_id: str | None = None,
    pricing_snapshot_date: str | None = None,
) -> dict[str, Any]:
    """Score a hosted provider/model through the standard submission contract."""
    if mode not in {"text", "audio"}:
        raise ValueError("mode must be 'text' or 'audio'")
    bench = OpenVoiceCSBench.load(scenario_path)
    fn = build_provider_agent(provider_spec)
    metadata = provider_metadata(
        provider_spec,
        input_modality="audio" if mode == "audio" else "text",
        pricing_profile_id=pricing_profile_id,
        pricing_snapshot_date=pricing_snapshot_date,
    )
    if mode == "audio":
        return bench.score_audio_manifest(
            fn,
            manifest_path=audio_manifest_path,
            max_variants=max_items,
            trials=trials,
            track=track,
            model_metadata=metadata,
        )
    return bench.score_agent(
        fn,
        max_scenarios=max_items,
        trials=trials,
        track=track,
        model_metadata=metadata,
    )


def build_submission_card_from_file(
    report_path: str | Path,
    *,
    submitter_name: str | None = None,
    organization: str | None = None,
    contact: str | None = None,
    repository_url: str | None = None,
    license_id: str | None = None,
    training_data_statement: str = "not_provided",
    safety_statement: str = "not_provided",
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    """Build a submission disclosure card from a saved report JSON."""
    path = Path(report_path)
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    return build_submission_card(
        report,
        report_path=path,
        submitter_name=submitter_name,
        organization=organization,
        contact=contact,
        repository_url=repository_url,
        license_id=license_id,
        training_data_statement=training_data_statement,
        safety_statement=safety_statement,
        limitations=limitations,
    )


def build_submission_card(
    report: dict[str, Any],
    *,
    report_path: str | Path | None = None,
    submitter_name: str | None = None,
    organization: str | None = None,
    contact: str | None = None,
    repository_url: str | None = None,
    license_id: str | None = None,
    training_data_statement: str = "not_provided",
    safety_statement: str = "not_provided",
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    """Build a machine-readable submission card for leaderboard intake."""
    metadata = report.get("model_metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    report_entry = _file_entry(report_path) if report_path is not None else None
    system_name = (
        metadata.get("display_name")
        or metadata.get("model_id")
        or metadata.get("agent")
        or metadata.get("submission_spec")
        or "unknown"
    )
    input_modality = metadata.get("input_modality")
    if not input_modality:
        input_modality = "audio" if report.get("evaluation_mode") == "audio_manifest" else "text"
    return {
        "benchmark": "OpenVoiceCS-Bench Submission Card",
        "card_version": "0.1.0",
        "generated_at": time.strftime("%Y-%m-%d"),
        "submitter": {
            "name": submitter_name,
            "organization": organization,
            "contact": contact,
        },
        "system": {
            "name": str(system_name),
            "provider": metadata.get("provider") or "not_disclosed",
            "model_id": metadata.get("model_id"),
            "submission_spec": metadata.get("submission_spec"),
            "pipeline_type": metadata.get("pipeline_type") or "unknown",
            "input_modality": input_modality,
            "repository_url": repository_url,
            "license": license_id,
        },
        "pricing": {
            "pricing_profile_id": metadata.get("pricing_profile_id")
            or metadata.get("pricing_profile"),
            "pricing_snapshot_date": metadata.get("pricing_snapshot_date"),
            "embedded_pricing_present": isinstance(metadata.get("pricing"), dict),
        },
        "evaluation": {
            "report": report_entry,
            "benchmark_version": report.get("benchmark_version"),
            "evaluation_mode": report.get("evaluation_mode", "text"),
            "num_scenarios": report.get("num_scenarios"),
            "num_audio_variants": report.get("num_audio_variants"),
            "num_trials_per_scenario": report.get("num_trials_per_scenario"),
            "overall_score": report.get("overall_score"),
            "pass_at_k": report.get("pass_at_k"),
            "pass_k": report.get("pass_k"),
            "mean_pass_rate": report.get("mean_pass_rate"),
            "conversation_experience_score": report.get("conversation_experience_score"),
        },
        "disclosures": {
            "training_data_statement": training_data_statement,
            "safety_statement": safety_statement,
            "limitations": limitations or [],
        },
    }


def validate_submission_card_file(path: str | Path) -> list[SubmissionCardIssue]:
    """Validate a saved submission-card JSON file."""
    with open(path, encoding="utf-8") as f:
        card = json.load(f)
    return validate_submission_card(card)


def load_submission_intake(
    path: str | Path = DEFAULT_SUBMISSION_INTAKE_PATH,
    *,
    base_dir: str | Path = ".",
) -> dict[str, Any]:
    """Load and validate an OpenVoiceCS official submission intake envelope."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        envelope = json.load(f)
    issues = validate_submission_intake(envelope, base_dir=base_dir)
    if issues:
        formatted = "\n".join(
            f"- {issue.item_id}::{issue.path}: {issue.message}"
            for issue in issues
        )
        raise ValueError(f"OpenVoiceCS submission intake validation failed:\n{formatted}")
    return envelope


def validate_submission_intake_file(
    path: str | Path = DEFAULT_SUBMISSION_INTAKE_PATH,
    *,
    base_dir: str | Path = ".",
) -> list[SubmissionIntakeIssue]:
    """Validate a saved official-submission intake envelope."""
    with open(path, encoding="utf-8") as f:
        envelope = json.load(f)
    return validate_submission_intake(envelope, base_dir=base_dir)


def validate_submission_intake(
    envelope: dict[str, Any],
    *,
    base_dir: str | Path = ".",
) -> list[SubmissionIntakeIssue]:
    """Return all official-submission intake envelope contract issues."""
    issues: list[SubmissionIntakeIssue] = []
    if not isinstance(envelope, dict):
        return [SubmissionIntakeIssue("<submission-intake>", "<root>", "must be an object")]
    for field in (
        "benchmark",
        "intake_version",
        "benchmark_version",
        "submission_id",
        "system_id",
        "status",
        "official_submission",
        "artifacts",
        "review",
    ):
        if field not in envelope:
            issues.append(SubmissionIntakeIssue("<submission-intake>", field, "missing required field"))
    if issues:
        return issues

    item_id = str(envelope.get("submission_id") or "<submission-intake>")
    if envelope.get("benchmark") != "OpenVoiceCS-Bench Submission Intake":
        issues.append(
            SubmissionIntakeIssue(
                item_id,
                "benchmark",
                "must be OpenVoiceCS-Bench Submission Intake",
            )
        )
    for field in ("intake_version", "benchmark_version", "submission_id", "system_id"):
        if not _non_empty_string(envelope.get(field)):
            issues.append(SubmissionIntakeIssue(item_id, field, "must be a non-empty string"))
    status = envelope.get("status")
    if status not in SUBMISSION_STATUSES:
        issues.append(
            SubmissionIntakeIssue(
                item_id,
                "status",
                f"must be one of: {', '.join(sorted(SUBMISSION_STATUSES))}",
            )
        )
    if not isinstance(envelope.get("official_submission"), bool):
        issues.append(SubmissionIntakeIssue(item_id, "official_submission", "must be boolean"))
    if status == "reference_fixture" and envelope.get("official_submission") is not False:
        issues.append(
            SubmissionIntakeIssue(
                item_id,
                "official_submission",
                "reference fixtures cannot be official submissions",
            )
        )

    artifacts = envelope.get("artifacts")
    loaded_artifacts: dict[str, dict[str, Any] | None] = {}
    if not isinstance(artifacts, dict):
        issues.append(SubmissionIntakeIssue(item_id, "artifacts", "must be an object"))
    else:
        base_path = Path(base_dir)
        for name, entry in artifacts.items():
            if name not in SUBMISSION_REQUIRED_ARTIFACTS:
                issues.append(SubmissionIntakeIssue(item_id, f"artifacts.{name}", "unknown artifact key"))
                continue
            loaded_artifacts[name] = _load_intake_file_entry(
                issues,
                entry,
                f"artifacts.{name}",
                item_id,
                base_path,
            )
        for name in SUBMISSION_REQUIRED_ARTIFACTS:
            if name not in artifacts:
                issues.append(SubmissionIntakeIssue(item_id, f"artifacts.{name}", "missing required artifact"))

    _validate_intake_artifact_contracts(issues, item_id, loaded_artifacts)
    _validate_intake_review(
        issues,
        item_id,
        envelope.get("review"),
        official=envelope.get("official_submission") is True,
        status=status,
    )
    if status == "official":
        _validate_official_submission_intake(issues, item_id, envelope, loaded_artifacts)
    return issues


def submission_intake_stats(envelope: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize official-submission intake evidence for release audits."""
    if not isinstance(envelope, dict):
        return {"present": False, "num_artifacts": 0}
    artifacts = envelope.get("artifacts", {})
    artifacts = artifacts if isinstance(artifacts, dict) else {}
    return {
        "present": True,
        "intake_version": envelope.get("intake_version"),
        "benchmark_version": envelope.get("benchmark_version"),
        "submission_id": envelope.get("submission_id"),
        "system_id": envelope.get("system_id"),
        "status": envelope.get("status"),
        "official_submission": envelope.get("official_submission"),
        "num_artifacts": len(artifacts),
        "required_artifacts_present": sum(
            1 for name in SUBMISSION_REQUIRED_ARTIFACTS if name in artifacts
        ),
    }


def validate_submission_card(card: dict[str, Any]) -> list[SubmissionCardIssue]:
    """Validate a submission card contract."""
    issues: list[SubmissionCardIssue] = []
    if not isinstance(card, dict):
        return [SubmissionCardIssue("<submission-card>", "<root>", "must be an object")]
    for field in (
        "benchmark",
        "card_version",
        "submitter",
        "system",
        "pricing",
        "evaluation",
        "disclosures",
    ):
        if field not in card:
            issues.append(SubmissionCardIssue("<submission-card>", field, "missing required field"))
    if issues:
        return issues
    if card.get("benchmark") != "OpenVoiceCS-Bench Submission Card":
        issues.append(
            SubmissionCardIssue(
                "<submission-card>",
                "benchmark",
                "must be OpenVoiceCS-Bench Submission Card",
            )
        )
    _validate_submitter(issues, card.get("submitter"))
    _validate_system(issues, card.get("system"))
    _validate_pricing(issues, card.get("pricing"))
    _validate_evaluation(issues, card.get("evaluation"))
    _validate_disclosures(issues, card.get("disclosures"))
    return issues


def _validate_submitter(
    issues: list[SubmissionCardIssue],
    submitter: Any,
) -> None:
    if not isinstance(submitter, dict):
        issues.append(SubmissionCardIssue("<submission-card>", "submitter", "must be an object"))
        return
    for field in ("name", "organization", "contact"):
        value = submitter.get(field)
        if value is not None and not isinstance(value, str):
            issues.append(
                SubmissionCardIssue(
                    "<submission-card>",
                    f"submitter.{field}",
                    "must be a string or null",
                )
            )


def _validate_system(
    issues: list[SubmissionCardIssue],
    system: Any,
) -> None:
    if not isinstance(system, dict):
        issues.append(SubmissionCardIssue("<submission-card>", "system", "must be an object"))
        return
    if not _non_empty_string(system.get("name")):
        issues.append(
            SubmissionCardIssue(
                "<submission-card>",
                "system.name",
                "must be a non-empty string",
            )
        )
    if (
        not _non_empty_string(system.get("model_id"))
        and not _non_empty_string(system.get("submission_spec"))
    ):
        issues.append(
            SubmissionCardIssue(
                "<submission-card>",
                "system.model_id",
                "must include model_id or submission_spec",
            )
        )
    pipeline_type = system.get("pipeline_type")
    if pipeline_type not in PIPELINE_TYPES:
        issues.append(
            SubmissionCardIssue(
                "<submission-card>",
                "system.pipeline_type",
                f"must be one of: {', '.join(sorted(PIPELINE_TYPES))}",
            )
        )
    input_modality = system.get("input_modality")
    if input_modality not in INPUT_MODALITIES:
        issues.append(
            SubmissionCardIssue(
                "<submission-card>",
                "system.input_modality",
                f"must be one of: {', '.join(sorted(INPUT_MODALITIES))}",
            )
        )
    for field in ("provider", "repository_url", "license"):
        value = system.get(field)
        if value is not None and not isinstance(value, str):
            issues.append(
                SubmissionCardIssue(
                    "<submission-card>",
                    f"system.{field}",
                    "must be a string or null",
                )
            )


def _validate_pricing(
    issues: list[SubmissionCardIssue],
    pricing: Any,
) -> None:
    if not isinstance(pricing, dict):
        issues.append(SubmissionCardIssue("<submission-card>", "pricing", "must be an object"))
        return
    for field in ("pricing_profile_id", "pricing_snapshot_date"):
        value = pricing.get(field)
        if value is not None and not isinstance(value, str):
            issues.append(
                SubmissionCardIssue(
                    "<submission-card>",
                    f"pricing.{field}",
                    "must be a string or null",
                )
            )
    if (
        "embedded_pricing_present" not in pricing
        or not isinstance(pricing.get("embedded_pricing_present"), bool)
    ):
        issues.append(
            SubmissionCardIssue(
                "<submission-card>",
                "pricing.embedded_pricing_present",
                "must be boolean",
            )
        )


def _validate_evaluation(
    issues: list[SubmissionCardIssue],
    evaluation: Any,
) -> None:
    if not isinstance(evaluation, dict):
        issues.append(SubmissionCardIssue("<submission-card>", "evaluation", "must be an object"))
        return
    for field in ("num_scenarios", "num_trials_per_scenario"):
        value = evaluation.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            issues.append(
                SubmissionCardIssue(
                    "<submission-card>",
                    f"evaluation.{field}",
                    "must be a nonnegative integer",
                )
            )
    audio_variants = evaluation.get("num_audio_variants")
    if audio_variants is not None and (
        isinstance(audio_variants, bool)
        or not isinstance(audio_variants, int)
        or audio_variants < 0
    ):
        issues.append(
            SubmissionCardIssue(
                "<submission-card>",
                "evaluation.num_audio_variants",
                "must be a nonnegative integer or null",
            )
        )
    for field in ("overall_score",):
        _validate_number_range(
            issues,
            evaluation.get(field),
            f"evaluation.{field}",
            low=0,
            high=100,
        )
    for field in ("pass_at_k", "pass_k", "mean_pass_rate", "conversation_experience_score"):
        _validate_number_range(
            issues,
            evaluation.get(field),
            f"evaluation.{field}",
            low=0,
            high=1,
            allow_none=True,
        )
    report = evaluation.get("report")
    if report is not None:
        _validate_file_entry(issues, report, "evaluation.report")


def _validate_disclosures(
    issues: list[SubmissionCardIssue],
    disclosures: Any,
) -> None:
    if not isinstance(disclosures, dict):
        issues.append(SubmissionCardIssue("<submission-card>", "disclosures", "must be an object"))
        return
    for field in ("training_data_statement", "safety_statement"):
        if not _non_empty_string(disclosures.get(field)):
            issues.append(
                SubmissionCardIssue(
                    "<submission-card>",
                    f"disclosures.{field}",
                    "must be a non-empty string",
                )
            )
    limitations = disclosures.get("limitations")
    if not isinstance(limitations, list):
        issues.append(
            SubmissionCardIssue(
                "<submission-card>",
                "disclosures.limitations",
                "must be a list",
            )
        )
    elif not all(isinstance(item, str) for item in limitations):
        issues.append(
            SubmissionCardIssue(
                "<submission-card>",
                "disclosures.limitations",
                "must contain only strings",
            )
        )


def _validate_file_entry(
    issues: list[SubmissionCardIssue],
    entry: Any,
    path: str,
) -> None:
    if not isinstance(entry, dict):
        issues.append(SubmissionCardIssue("<submission-card>", path, "must be an object"))
        return
    for field in ("path", "sha256", "bytes"):
        if field not in entry:
            issues.append(
                SubmissionCardIssue(
                    "<submission-card>",
                    f"{path}.{field}",
                    "missing required field",
                )
            )
    if "sha256" in entry and not _looks_sha256(entry.get("sha256")):
        issues.append(
            SubmissionCardIssue(
                "<submission-card>",
                f"{path}.sha256",
                "must be a SHA-256 hex digest",
            )
        )
    if "bytes" in entry and (
        isinstance(entry.get("bytes"), bool)
        or not isinstance(entry.get("bytes"), int)
        or entry["bytes"] < 0
    ):
        issues.append(
            SubmissionCardIssue(
                "<submission-card>",
                f"{path}.bytes",
                "must be a nonnegative integer",
            )
        )


def _validate_number_range(
    issues: list[SubmissionCardIssue],
    value: Any,
    path: str,
    *,
    low: float,
    high: float,
    allow_none: bool = False,
) -> None:
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        issues.append(SubmissionCardIssue("<submission-card>", path, "must be numeric"))
    elif value < low or value > high:
        issues.append(
            SubmissionCardIssue(
                "<submission-card>",
                path,
                f"must be between {low} and {high}",
            )
        )


def _file_entry(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    data = resolved.read_bytes()
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _looks_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _load_intake_file_entry(
    issues: list[SubmissionIntakeIssue],
    entry: Any,
    path: str,
    item_id: str,
    base_dir: Path,
) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        issues.append(SubmissionIntakeIssue(item_id, path, "must be an object"))
        return None
    for field in ("path", "sha256", "bytes"):
        if field not in entry:
            issues.append(SubmissionIntakeIssue(item_id, f"{path}.{field}", "missing required field"))
    raw_path = entry.get("path")
    if not _non_empty_string(raw_path):
        issues.append(SubmissionIntakeIssue(item_id, f"{path}.path", "must be a non-empty string"))
        return None
    if "sha256" in entry and not _looks_sha256(entry.get("sha256")):
        issues.append(SubmissionIntakeIssue(item_id, f"{path}.sha256", "must be a SHA-256 hex digest"))
    if "bytes" in entry and (
        isinstance(entry.get("bytes"), bool)
        or not isinstance(entry.get("bytes"), int)
        or entry["bytes"] < 0
    ):
        issues.append(SubmissionIntakeIssue(item_id, f"{path}.bytes", "must be a nonnegative integer"))

    resolved = Path(str(raw_path))
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    if not resolved.exists():
        issues.append(SubmissionIntakeIssue(item_id, f"{path}.path", "file does not exist"))
        return None
    data = resolved.read_bytes()
    if entry.get("sha256") != hashlib.sha256(data).hexdigest():
        issues.append(SubmissionIntakeIssue(item_id, f"{path}.sha256", "does not match file contents"))
    if entry.get("bytes") != len(data):
        issues.append(SubmissionIntakeIssue(item_id, f"{path}.bytes", "does not match file size"))
    try:
        loaded = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(SubmissionIntakeIssue(item_id, path, f"invalid JSON: {exc.msg}"))
        return None
    if not isinstance(loaded, dict):
        issues.append(SubmissionIntakeIssue(item_id, path, "must contain a JSON object"))
        return None
    loaded["_openvoicecs_file_parent"] = str(resolved.parent)
    loaded["_openvoicecs_file_sha256"] = hashlib.sha256(data).hexdigest()
    loaded["_openvoicecs_file_bytes"] = len(data)
    return loaded


def _validate_intake_artifact_contracts(
    issues: list[SubmissionIntakeIssue],
    item_id: str,
    loaded_artifacts: dict[str, dict[str, Any] | None],
) -> None:
    card = loaded_artifacts.get("submission_card")
    if card is not None:
        issues.extend(
            SubmissionIntakeIssue(item_id, f"artifacts.submission_card.{issue.path}", issue.message)
            for issue in validate_submission_card(card)
        )
    report = loaded_artifacts.get("report")
    if report is not None:
        from src.evaluation.benchmark.openvoicecs import validate_report

        issues.extend(
            SubmissionIntakeIssue(item_id, f"artifacts.report.{issue.path}", issue.message)
            for issue in validate_report(report)
        )
    run_manifest = loaded_artifacts.get("run_manifest")
    if run_manifest is not None:
        from src.evaluation.benchmark.run_manifest import validate_run_manifest

        issues.extend(
            SubmissionIntakeIssue(item_id, f"artifacts.run_manifest.{issue.path}", issue.message)
            for issue in validate_run_manifest(run_manifest)
        )
    release_bundle = loaded_artifacts.get("release_bundle")
    if release_bundle is not None:
        from src.evaluation.benchmark.release_bundle import validate_frontier_release_bundle

        release_bundle_base = release_bundle.get("_openvoicecs_file_parent", ".")
        issues.extend(
            SubmissionIntakeIssue(item_id, f"artifacts.release_bundle.{issue.path}", issue.message)
            for issue in validate_frontier_release_bundle(
                release_bundle,
                base_dir=release_bundle_base,
            )
        )
    external_systems = loaded_artifacts.get("external_systems_registry")
    if external_systems is not None:
        from src.evaluation.benchmark.external_systems import validate_external_systems_registry

        issues.extend(
            SubmissionIntakeIssue(item_id, f"artifacts.external_systems_registry.{issue.path}", issue.message)
            for issue in validate_external_systems_registry(external_systems)
        )
    judge_package = loaded_artifacts.get("judge_annotation_package")
    if judge_package is not None:
        from src.evaluation.benchmark.judging import validate_judge_annotation_package

        issues.extend(
            SubmissionIntakeIssue(item_id, f"artifacts.judge_annotation_package.{issue.path}", issue.message)
            for issue in validate_judge_annotation_package(judge_package)
        )
    claims = loaded_artifacts.get("leaderboard_claims")
    if claims is not None:
        from src.evaluation.benchmark.claims import validate_claims_manifest

        issues.extend(
            SubmissionIntakeIssue(item_id, f"artifacts.leaderboard_claims.{issue.path}", issue.message)
            for issue in validate_claims_manifest(claims)
        )
    _validate_intake_cross_artifact_links(issues, item_id, loaded_artifacts)


def _validate_intake_cross_artifact_links(
    issues: list[SubmissionIntakeIssue],
    item_id: str,
    loaded_artifacts: dict[str, dict[str, Any] | None],
) -> None:
    card = loaded_artifacts.get("submission_card")
    report = loaded_artifacts.get("report")
    report_entry = (card or {}).get("evaluation", {}).get("report") if isinstance(card, dict) else None
    if isinstance(report_entry, dict) and report is not None:
        if report_entry.get("sha256") != report.get("_openvoicecs_file_sha256"):
            issues.append(
                SubmissionIntakeIssue(
                    item_id,
                    "artifacts.submission_card.evaluation.report.sha256",
                    "must match intake report artifact hash",
                )
            )
        if report_entry.get("bytes") != report.get("_openvoicecs_file_bytes"):
            issues.append(
                SubmissionIntakeIssue(
                    item_id,
                    "artifacts.submission_card.evaluation.report.bytes",
                    "must match intake report artifact size",
                )
            )
        if card.get("evaluation", {}).get("overall_score") != report.get("overall_score"):
            issues.append(
                SubmissionIntakeIssue(
                    item_id,
                    "artifacts.submission_card.evaluation.overall_score",
                    "must match report overall_score",
                )
            )
        if card.get("evaluation", {}).get("num_scenarios") != report.get("num_scenarios"):
            issues.append(
                SubmissionIntakeIssue(
                    item_id,
                    "artifacts.submission_card.evaluation.num_scenarios",
                    "must match report num_scenarios",
                )
            )
        if card.get("evaluation", {}).get("benchmark_version") != report.get("benchmark_version"):
            issues.append(
                SubmissionIntakeIssue(
                    item_id,
                    "artifacts.submission_card.evaluation.benchmark_version",
                    "must match report benchmark_version",
                )
            )
        if report_entry.get("sha256") is None:
            return
        if not _looks_sha256(report_entry.get("sha256")):
            issues.append(
                SubmissionIntakeIssue(
                    item_id,
                    "artifacts.submission_card.evaluation.report.sha256",
                    "must be a SHA-256 hex digest",
                )
            )


def _validate_intake_review(
    issues: list[SubmissionIntakeIssue],
    item_id: str,
    review: Any,
    *,
    official: bool,
    status: Any,
) -> None:
    if not isinstance(review, dict):
        issues.append(SubmissionIntakeIssue(item_id, "review", "must be an object"))
        return
    for field in ("review_status", "reviewed_by", "reviewed_at", "evidence_level"):
        if field not in review:
            issues.append(SubmissionIntakeIssue(item_id, f"review.{field}", "missing required field"))
    if not isinstance(review.get("reviewed_by"), list):
        issues.append(SubmissionIntakeIssue(item_id, "review.reviewed_by", "must be a list"))
    elif official and not review["reviewed_by"]:
        issues.append(SubmissionIntakeIssue(item_id, "review.reviewed_by", "official submissions need reviewers"))
    for field in ("review_status", "evidence_level"):
        if field in review and not _non_empty_string(review.get(field)):
            issues.append(SubmissionIntakeIssue(item_id, f"review.{field}", "must be a non-empty string"))
    if review.get("review_status") == "accepted" and status not in {"official", "reference_fixture"}:
        issues.append(
            SubmissionIntakeIssue(
                item_id,
                "review.review_status",
                "accepted review requires official or reference_fixture status",
            )
        )
    if official and review.get("evidence_level") == "reference_fixture":
        issues.append(
            SubmissionIntakeIssue(
                item_id,
                "review.evidence_level",
                "official submissions cannot use reference_fixture evidence",
            )
        )


def _validate_official_submission_intake(
    issues: list[SubmissionIntakeIssue],
    item_id: str,
    envelope: dict[str, Any],
    loaded_artifacts: dict[str, dict[str, Any] | None],
) -> None:
    if envelope.get("official_submission") is not True:
        issues.append(SubmissionIntakeIssue(item_id, "official_submission", "official status requires official_submission true"))
    for name in SUBMISSION_REQUIRED_ARTIFACTS:
        if loaded_artifacts.get(name) is None:
            issues.append(SubmissionIntakeIssue(item_id, f"artifacts.{name}", "official submissions must include valid artifact"))
    card = loaded_artifacts.get("submission_card") or {}
    if card.get("disclosures", {}).get("training_data_statement") in {None, "not_provided"}:
        issues.append(
            SubmissionIntakeIssue(
                item_id,
                "artifacts.submission_card.disclosures.training_data_statement",
                "official submissions must disclose training-data policy",
            )
        )
    if card.get("disclosures", {}).get("safety_statement") in {None, "not_provided"}:
        issues.append(
            SubmissionIntakeIssue(
                item_id,
                "artifacts.submission_card.disclosures.safety_statement",
                "official submissions must disclose safety policy",
            )
        )
