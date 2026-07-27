"""Tests for provider-neutral OpenVoiceCS endpoint submissions."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.evaluation.benchmark.external_endpoint import (
    build_external_endpoint_payload,
    external_endpoint_contract_stats,
    load_external_endpoint_contract,
    normalize_external_endpoint_response,
    score_external_endpoint,
    validate_external_endpoint_contract,
    validate_external_endpoint_contract_file,
)
from src.evaluation.benchmark.openvoicecs import OpenVoiceCSBench, oracle_agent, validate_report


def test_external_endpoint_contract_validates():
    contract = load_external_endpoint_contract()

    assert validate_external_endpoint_contract(contract) == []
    assert validate_external_endpoint_contract_file() == []
    stats = external_endpoint_contract_stats(contract)
    assert stats["transport"] == "http_json"
    assert stats["oracle_redaction_required"] is True
    assert stats["official_endpoint_runs_supported"] is True


def test_external_endpoint_contract_rejects_oracle_leakage_and_weak_security():
    contract = load_external_endpoint_contract()
    contract["request_schema"]["forbidden_fields"] = ["reference_trace"]
    contract["security"]["forbid_oracle_fields"] = False
    contract["response_schema"]["required_fields"] = ["messages"]

    messages = {
        (issue.path, issue.message)
        for issue in validate_external_endpoint_contract(contract)
    }

    assert (
        "request_schema.forbidden_fields",
        "must forbid oracle and initial_state",
    ) in messages
    assert ("security.forbid_oracle_fields", "must be true") in messages
    assert ("response_schema.required_fields", "must include tool_calls") in messages
    assert ("response_schema.required_fields", "must include events") in messages


def test_external_endpoint_payload_redacts_hidden_scenario_fields():
    scenario = OpenVoiceCSBench.load().scenarios[0]

    payload = build_external_endpoint_payload(scenario, trial_index=2, run_id="run-123")

    assert payload["protocol"] == "openvoicecs.endpoint.v1"
    assert payload["scenario_id"] == scenario["id"]
    assert payload["run_id"] == "run-123"
    assert "oracle" not in payload["scenario"]
    assert "initial_state" not in payload["scenario"]
    assert "oracle" not in json.dumps(payload)
    assert payload["tools"] == scenario["tools"]


def test_normalize_external_endpoint_response_requires_trace_contract():
    trace = normalize_external_endpoint_response(
        {
            "trace": {
                "messages": [{"role": "agent", "text": "Done."}],
                "tool_calls": [],
                "events": [],
                "latency_ms": 123,
            }
        }
    )

    assert trace["messages"][0]["text"] == "Done."
    assert trace["latency_ms"] == 123

    with pytest.raises(ValueError, match="tool_calls"):
        normalize_external_endpoint_response({"messages": [], "events": []})
    with pytest.raises(ValueError, match="unsupported"):
        normalize_external_endpoint_response(
            {"messages": [], "tool_calls": [], "events": [], "oracle": {}}
        )


def test_score_external_endpoint_against_loopback_http_server():
    bench = OpenVoiceCSBench.load()
    scenario_map = {scenario["id"]: scenario for scenario in bench.scenarios}
    seen_payloads: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - stdlib callback name
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            seen_payloads.append(payload)
            assert "oracle" not in payload["scenario"]
            assert "initial_state" not in payload["scenario"]
            trace = oracle_agent(scenario_map[payload["scenario_id"]], payload["trial_index"])
            response = {
                "messages": trace["messages"],
                "tool_calls": trace["tool_calls"],
                "events": trace["events"],
                "latency_ms": trace["latency_ms"],
                "cost_usd": 0.01,
            }
            body = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002 - stdlib signature
            return

    try:
        server = HTTPServer(("127.0.0.1", 0), Handler)
    except PermissionError:
        pytest.skip("local socket binding is not permitted in this environment")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/run"
        report = score_external_endpoint(
            endpoint,
            scenario_path="data/openvoicecs/scenarios_v0.1.json",
            audio_manifest_path="data/openvoicecs/audio_manifest_v0.1.json",
            max_items=1,
            trials=1,
            timeout_seconds=5,
            model_metadata={"display_name": "loopback-endpoint"},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert validate_report(report) == []
    assert report["pass_k"] == 1.0
    assert report["model_metadata"]["transport"] == "http_json"
    assert seen_payloads[0]["requested_trace_fields"] == [
        "messages",
        "tool_calls",
        "events",
        "latency",
        "usage",
        "cost_usd",
    ]
