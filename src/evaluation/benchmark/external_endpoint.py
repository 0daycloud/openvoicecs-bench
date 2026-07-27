"""Provider-neutral HTTP endpoint contract for OpenVoiceCS submissions."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

DEFAULT_EXTERNAL_ENDPOINT_CONTRACT_PATH = Path(
    "data/openvoicecs/external_endpoint_contract_v0.1.json"
)
CONTRACT_TRANSPORTS = {"http_json"}
SENSITIVE_SCENARIO_FIELDS = {
    "oracle",
    "initial_state",
    "expected_state",
    "reference_trace",
    "judge_notes",
    "sealed_metadata",
}
TRACE_FIELDS = {"messages", "tool_calls", "events", "claims", "usage", "cost_usd", "latency", "latency_ms"}


@dataclass(frozen=True)
class ExternalEndpointIssue:
    """Structured external endpoint contract validation issue."""

    item_id: str
    path: str
    message: str


def load_external_endpoint_contract(
    path: str | Path = DEFAULT_EXTERNAL_ENDPOINT_CONTRACT_PATH,
) -> dict[str, Any]:
    """Load and validate the endpoint contract manifest."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        contract = json.load(f)
    issues = validate_external_endpoint_contract(contract)
    if issues:
        formatted = "\n".join(
            f"- {issue.item_id}::{issue.path}: {issue.message}"
            for issue in issues
        )
        raise ValueError(f"OpenVoiceCS external endpoint contract validation failed:\n{formatted}")
    return contract


def validate_external_endpoint_contract_file(
    path: str | Path = DEFAULT_EXTERNAL_ENDPOINT_CONTRACT_PATH,
) -> list[ExternalEndpointIssue]:
    """Validate a saved endpoint contract manifest."""
    with open(path, encoding="utf-8") as f:
        contract = json.load(f)
    return validate_external_endpoint_contract(contract)


def validate_external_endpoint_contract(contract: dict[str, Any]) -> list[ExternalEndpointIssue]:
    """Return all endpoint contract issues."""
    issues: list[ExternalEndpointIssue] = []
    if not isinstance(contract, dict):
        return [ExternalEndpointIssue("<external_endpoint>", "<root>", "must be an object")]
    for field in (
        "name",
        "version",
        "benchmark_version",
        "transport",
        "request_schema",
        "response_schema",
        "timeouts",
        "security",
        "official_run_requirements",
    ):
        if field not in contract:
            issues.append(ExternalEndpointIssue("<external_endpoint>", field, "missing required field"))
    if issues:
        return issues

    if contract.get("name") != "OpenVoiceCS External Endpoint Contract":
        issues.append(
            ExternalEndpointIssue(
                "<external_endpoint>",
                "name",
                "must be OpenVoiceCS External Endpoint Contract",
            )
        )
    for field in ("version", "benchmark_version"):
        if not _non_empty_string(contract.get(field)):
            issues.append(ExternalEndpointIssue("<external_endpoint>", field, "must be a non-empty string"))
    if contract.get("transport") not in CONTRACT_TRANSPORTS:
        issues.append(
            ExternalEndpointIssue(
                "<external_endpoint>",
                "transport",
                f"must be one of: {', '.join(sorted(CONTRACT_TRANSPORTS))}",
            )
        )
    _validate_schema(issues, contract.get("request_schema"), path="request_schema")
    _validate_schema(issues, contract.get("response_schema"), path="response_schema")
    _validate_timeouts(issues, contract.get("timeouts"))
    _validate_security(issues, contract.get("security"))
    _validate_official_requirements(issues, contract.get("official_run_requirements"))
    return issues


def external_endpoint_contract_stats(contract: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize endpoint contract evidence for release audits."""
    if not isinstance(contract, dict):
        return {"present": False}
    response_schema = contract.get("response_schema")
    response_schema = response_schema if isinstance(response_schema, dict) else {}
    return {
        "present": True,
        "version": contract.get("version"),
        "benchmark_version": contract.get("benchmark_version"),
        "transport": contract.get("transport"),
        "required_request_fields": len(
            (contract.get("request_schema") or {}).get("required_fields") or []
        ),
        "required_response_fields": len(response_schema.get("required_fields") or []),
        "oracle_redaction_required": (
            (contract.get("security") or {}).get("forbid_oracle_fields") is True
        ),
        "official_endpoint_runs_supported": (
            (contract.get("official_run_requirements") or {}).get("requires_live_endpoint_run")
            is True
        ),
    }


def build_external_endpoint_payload(
    scenario: dict[str, Any],
    trial_index: int,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build the redacted HTTP JSON request body for one benchmark trial."""
    safe_scenario = _redact_scenario(scenario)
    audio_variant = safe_scenario.get("audio_variant")
    return {
        "protocol": "openvoicecs.endpoint.v1",
        "benchmark": "OpenVoiceCS-Bench",
        "benchmark_version": "0.1.0",
        "run_id": run_id,
        "scenario_id": scenario.get("id"),
        "base_scenario_id": scenario.get("base_scenario_id"),
        "trial_index": trial_index,
        "input_modality": scenario.get("input_modality", "text"),
        "track": scenario.get("track"),
        "domain": scenario.get("domain"),
        "difficulty": scenario.get("difficulty"),
        "user_utterance": _scenario_user_utterance(scenario),
        "audio_variant": audio_variant,
        "scenario": safe_scenario,
        "tools": deepcopy(scenario.get("tools") or []),
        "requested_trace_fields": ["messages", "tool_calls", "events", "latency", "usage", "cost_usd"],
    }


def build_external_endpoint_agent(
    endpoint_url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout_seconds: float = 30.0,
    run_id: str | None = None,
) -> Any:
    """Return an OpenVoiceCS agent callable backed by an HTTP JSON endpoint."""
    if not _non_empty_string(endpoint_url):
        raise ValueError("endpoint_url must be a non-empty string")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")
    safe_headers = {"Content-Type": "application/json", "Accept": "application/json"}
    for key, value in (headers or {}).items():
        if not _non_empty_string(key) or not isinstance(value, str):
            raise ValueError("headers must be a mapping of non-empty string keys to string values")
        safe_headers[key] = value

    def _agent(scenario: dict[str, Any], trial_index: int) -> dict[str, Any]:
        payload = build_external_endpoint_payload(scenario, trial_index, run_id=run_id)
        started = time.perf_counter()
        body = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            endpoint_url,
            data=body,
            headers=safe_headers,
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=timeout_seconds) as response:
                response_body = response.read()
        except urllib_error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"endpoint returned HTTP {exc.code}: {details[:500]}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"endpoint request failed: {exc.reason}") from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        try:
            decoded = json.loads(response_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("endpoint response must be valid JSON") from exc
        trace = normalize_external_endpoint_response(decoded)
        trace.setdefault("latency", {})
        if isinstance(trace["latency"], dict):
            trace["latency"].setdefault("endpoint_round_trip_ms", round(elapsed_ms, 3))
        return trace

    return _agent


def normalize_external_endpoint_response(response: Any) -> dict[str, Any]:
    """Validate and normalize an endpoint response into an OpenVoiceCS trace."""
    if not isinstance(response, dict):
        raise TypeError("endpoint response must be a JSON object")
    trace = response.get("trace") if isinstance(response.get("trace"), dict) else response
    if not isinstance(trace, dict):
        raise TypeError("endpoint trace must be a JSON object")
    unknown = sorted(set(trace) - TRACE_FIELDS - {"metadata", "raw_response"})
    if unknown:
        raise ValueError(f"endpoint trace has unsupported field(s): {', '.join(unknown)}")
    messages = trace.get("messages")
    tool_calls = trace.get("tool_calls")
    events = trace.get("events")
    if not isinstance(messages, list):
        raise ValueError("endpoint trace.messages must be a list")
    if not isinstance(tool_calls, list):
        raise ValueError("endpoint trace.tool_calls must be a list")
    if not isinstance(events, list):
        raise ValueError("endpoint trace.events must be a list")
    normalized = {
        "messages": messages,
        "tool_calls": tool_calls,
        "events": events,
        "claims": trace.get("claims") or [],
        "usage": trace.get("usage") or {},
        "cost_usd": trace.get("cost_usd"),
        "latency": trace.get("latency") or {},
    }
    if trace.get("latency_ms") is not None:
        normalized["latency_ms"] = trace["latency_ms"]
    return normalized


def score_external_endpoint(
    endpoint_url: str,
    *,
    scenario_path: str | Path,
    audio_manifest_path: str | Path,
    mode: str = "text",
    max_items: int | None = None,
    trials: int = 1,
    track: str | None = None,
    headers: dict[str, str] | None = None,
    timeout_seconds: float = 30.0,
    run_id: str | None = None,
    model_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a provider-neutral HTTP endpoint over text scenarios or audio variants."""
    if mode not in {"text", "audio"}:
        raise ValueError("mode must be 'text' or 'audio'")
    from src.evaluation.benchmark.openvoicecs import OpenVoiceCSBench

    bench = OpenVoiceCSBench.load(scenario_path)
    agent = build_external_endpoint_agent(
        endpoint_url,
        headers=headers,
        timeout_seconds=timeout_seconds,
        run_id=run_id,
    )
    metadata = {
        "agent": endpoint_url,
        "endpoint_protocol": "openvoicecs.endpoint.v1",
        "endpoint_url": endpoint_url,
        "transport": "http_json",
        "provider": "external_endpoint",
        "input_modality": mode,
        **(model_metadata or {}),
    }
    if mode == "audio":
        return bench.score_audio_manifest(
            agent,
            manifest_path=audio_manifest_path,
            max_variants=max_items,
            trials=trials,
            track=track,
            model_metadata=metadata,
        )
    return bench.score_agent(
        agent,
        max_scenarios=max_items,
        trials=trials,
        track=track,
        model_metadata=metadata,
    )


def _validate_schema(
    issues: list[ExternalEndpointIssue],
    schema: Any,
    *,
    path: str,
) -> None:
    if not isinstance(schema, dict):
        issues.append(ExternalEndpointIssue("<external_endpoint>", path, "must be an object"))
        return
    required = schema.get("required_fields")
    optional = schema.get("optional_fields")
    if not isinstance(required, list) or not required:
        issues.append(
            ExternalEndpointIssue("<external_endpoint>", f"{path}.required_fields", "must be a non-empty list")
        )
    elif any(not _non_empty_string(field) for field in required):
        issues.append(
            ExternalEndpointIssue("<external_endpoint>", f"{path}.required_fields", "must contain strings")
        )
    if optional is not None and (
        not isinstance(optional, list)
        or any(not _non_empty_string(field) for field in optional)
    ):
        issues.append(
            ExternalEndpointIssue("<external_endpoint>", f"{path}.optional_fields", "must contain strings")
        )
    if path == "request_schema":
        required_set = set(required if isinstance(required, list) else [])
        for field in ("protocol", "scenario_id", "trial_index", "user_utterance", "scenario"):
            if field not in required_set:
                issues.append(
                    ExternalEndpointIssue("<external_endpoint>", f"{path}.required_fields", f"must include {field}")
                )
        forbidden = schema.get("forbidden_fields")
        if not isinstance(forbidden, list) or "oracle" not in forbidden or "initial_state" not in forbidden:
            issues.append(
                ExternalEndpointIssue(
                    "<external_endpoint>",
                    f"{path}.forbidden_fields",
                    "must forbid oracle and initial_state",
                )
            )
    if path == "response_schema":
        required_set = set(required if isinstance(required, list) else [])
        for field in ("messages", "tool_calls", "events"):
            if field not in required_set:
                issues.append(
                    ExternalEndpointIssue("<external_endpoint>", f"{path}.required_fields", f"must include {field}")
                )


def _validate_timeouts(issues: list[ExternalEndpointIssue], timeouts: Any) -> None:
    if not isinstance(timeouts, dict):
        issues.append(ExternalEndpointIssue("<external_endpoint>", "timeouts", "must be an object"))
        return
    request_timeout = timeouts.get("default_request_timeout_seconds")
    request_timeout_valid = (
        not isinstance(request_timeout, bool)
        and isinstance(request_timeout, (int, float))
        and request_timeout > 0
    )
    if not request_timeout_valid:
        issues.append(
            ExternalEndpointIssue(
                "<external_endpoint>",
                "timeouts.default_request_timeout_seconds",
                "must be numeric and > 0",
            )
        )
    max_timeout = timeouts.get("maximum_official_timeout_seconds")
    max_timeout_valid = (
        not isinstance(max_timeout, bool)
        and isinstance(max_timeout, (int, float))
        and (not request_timeout_valid or max_timeout >= request_timeout)
    )
    if not max_timeout_valid:
        issues.append(
            ExternalEndpointIssue(
                "<external_endpoint>",
                "timeouts.maximum_official_timeout_seconds",
                "must be numeric and >= default_request_timeout_seconds",
            )
        )


def _validate_security(issues: list[ExternalEndpointIssue], security: Any) -> None:
    if not isinstance(security, dict):
        issues.append(ExternalEndpointIssue("<external_endpoint>", "security", "must be an object"))
        return
    for field in (
        "forbid_oracle_fields",
        "forbid_initial_state",
        "redact_authoring_metadata",
        "allow_authorization_header",
        "never_log_authorization_header",
    ):
        if security.get(field) is not True:
            issues.append(ExternalEndpointIssue("<external_endpoint>", f"security.{field}", "must be true"))


def _validate_official_requirements(
    issues: list[ExternalEndpointIssue],
    requirements: Any,
) -> None:
    if not isinstance(requirements, dict):
        issues.append(ExternalEndpointIssue("<external_endpoint>", "official_run_requirements", "must be an object"))
        return
    for field in (
        "requires_live_endpoint_run",
        "requires_tls_for_remote_endpoint",
        "requires_hash_pinned_report",
        "requires_submission_intake",
        "requires_external_system_registry_entry",
        "requires_judged_report_for_leaderboard",
    ):
        if requirements.get(field) is not True:
            issues.append(
                ExternalEndpointIssue(
                    "<external_endpoint>",
                    f"official_run_requirements.{field}",
                    "must be true",
                )
            )
    min_trials = requirements.get("minimum_trials_per_scenario")
    if isinstance(min_trials, bool) or not isinstance(min_trials, int) or min_trials < 1:
        issues.append(
            ExternalEndpointIssue(
                "<external_endpoint>",
                "official_run_requirements.minimum_trials_per_scenario",
                "must be an integer >= 1",
            )
        )


def _redact_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    redacted = {
        key: deepcopy(value)
        for key, value in scenario.items()
        if key not in SENSITIVE_SCENARIO_FIELDS and not key.startswith("_")
    }
    for key in SENSITIVE_SCENARIO_FIELDS:
        redacted.pop(key, None)
    return redacted


def _scenario_user_utterance(scenario: dict[str, Any]) -> str:
    if _non_empty_string(scenario.get("user_utterance")):
        return str(scenario["user_utterance"])
    audio_variant = scenario.get("audio_variant")
    if isinstance(audio_variant, dict) and _non_empty_string(audio_variant.get("transcript")):
        return str(audio_variant["transcript"])
    for turn in scenario.get("conversation", []) or []:
        if isinstance(turn, dict) and turn.get("role") in {"user", "customer"}:
            text = turn.get("text") or turn.get("utterance")
            if _non_empty_string(text):
                return str(text)
    return ""


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
