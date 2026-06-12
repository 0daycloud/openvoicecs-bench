"""Tests for the reference realtime load harness."""

from __future__ import annotations

import asyncio
import json

import pytest

from src.evaluation.benchmark.frontier import build_frontier_report
from src.evaluation.benchmark.openvoicecs import OpenVoiceCSBench, oracle_agent, validate_report
from src.evaluation.benchmark.pricing import load_pricing_manifest
from src.evaluation.benchmark.realtime import (
    ReferenceRealtimeClient,
    WebRTCRealtimeClient,
    WebSocketRealtimeClient,
    build_realtime_request,
    builtin_realtime_agent,
    normalize_realtime_events,
    run_openvoicecs_realtime_load,
)


def test_build_realtime_request_includes_latency_origin_and_scenario():
    scenario = OpenVoiceCSBench.load().scenarios[0]

    request = build_realtime_request(
        scenario,
        trial_index=2,
        concurrency=10,
        seed=123,
        transport="in_process",
    )

    assert request["protocol"] == "openvoicecs.realtime.v1"
    assert request["scenario_id"] == scenario["id"]
    assert request["scenario"]["oracle"] == scenario["oracle"]
    assert request["events"][1]["type"] == "user.end_speech"
    assert request["events"][1]["t_ms"] == 0.0
    assert request["concurrency"] == 10


def test_webrtc_client_uses_canonical_transport_contract():
    client = WebRTCRealtimeClient("https://example.test/signaling")
    scenario = OpenVoiceCSBench.load().scenarios[0]

    request = build_realtime_request(
        scenario,
        trial_index=0,
        concurrency=1,
        seed=0,
        transport=client.transport,
    )

    assert client.transport == "webrtc"
    assert client.data_channel == "openvoicecs"
    assert request["protocol"] == "openvoicecs.realtime.v1"
    assert request["transport"] == "webrtc"


def test_websocket_client_round_trips_against_loopback_reference_endpoint():
    websockets = pytest.importorskip("websockets")

    async def exercise():
        agent = builtin_realtime_agent("oracle")

        async def handler(websocket):
            raw = await websocket.recv()
            request = json.loads(raw)
            trace = agent(request)
            for event in trace["events"]:
                await websocket.send(json.dumps(event))

        try:
            server = await websockets.serve(handler, "127.0.0.1", 0)
        except PermissionError:
            pytest.skip("local socket binding is not permitted in this environment")
        try:
            port = server.sockets[0].getsockname()[1]
            client = WebSocketRealtimeClient(f"ws://127.0.0.1:{port}")
            scenario = OpenVoiceCSBench.load().scenarios[0]
            trace = await client.run_call(
                scenario,
                trial_index=0,
                concurrency=1,
                seed=0,
            )
        finally:
            server.close()
            await server.wait_closed()
        return trace

    trace = asyncio.run(exercise())

    assert trace["latency"]["measurement"]["source"] == "event_stream"
    assert trace["latency"]["measurement"]["origin_event"] == "user.end_speech"
    assert trace["latency"]["measurement"]["first_audio_event"] == "tts.first_audio"
    assert trace["latency"]["measurement"]["last_audio_event"] == "agent.complete"
    assert trace["latency"]["v2v_ttfb_ms"] > 0
    assert trace["latency"]["v2v_last_byte_ms"] >= trace["latency"]["v2v_ttfb_ms"]


def test_normalize_realtime_events_maps_transport_events_to_trace():
    trace = normalize_realtime_events(
        [
            {"type": "asr.final", "t_ms": 40},
            {"type": "llm.first_token", "t_ms": 140},
            {"type": "tts.first_audio", "t_ms": 220},
            {"type": "agent.message", "text": "Done."},
            {"type": "tool.call", "name": "refund", "arguments": {"amount": 10}},
            {"type": "policy.event", "name": "identity_verified"},
            {"type": "usage", "usage": {"input_tokens": 100, "output_tokens": 20}},
            {"type": "barge_in.stop", "t_ms": 90},
            {"type": "barge_in.recovered", "t_ms": 180},
            {"type": "agent.complete", "t_ms": 520, "cost_usd": 0.01},
        ],
        measured_ms=600,
    )

    assert trace["latency"]["v2v_ttfb_ms"] == 220
    assert trace["latency"]["v2v_last_byte_ms"] == 520
    assert trace["latency"]["stage_latency_ms"]["asr_finalization_ms"] == 40
    assert trace["latency"]["stage_latency_ms"]["llm_ttft_ms"] == 140
    assert trace["latency"]["barge_in_stop_ms"] == 90
    assert trace["latency"]["interruption_recovery_ms"] == 180
    assert trace["latency"]["measurement"] == {
        "source": "event_stream",
        "origin_event": "user.end_speech",
        "origin_t_ms": 0.0,
        "first_audio_event": "tts.first_audio",
        "last_audio_event": "agent.complete",
        "barge_in_stop_event": "barge_in.stop",
        "interruption_recovery_event": "barge_in.recovered",
    }
    assert trace["messages"][0]["text"] == "Done."
    assert trace["tool_calls"][0]["name"] == "refund"
    assert trace["events"] == ["identity_verified"]
    assert trace["cost_usd"] == 0.01


def test_realtime_load_harness_scores_openvoicecs_under_concurrency():
    bench = OpenVoiceCSBench.load()

    async def agent(request):
        await asyncio.sleep(0.001)
        scenario = request["scenario"]
        trace = oracle_agent(scenario, request["trial_index"])
        ttfb = 200 + request["concurrency"]
        trace["cost_usd"] = 0.02
        trace["latency"] = {
            "v2v_ttfb_ms": ttfb,
            "v2v_last_byte_ms": ttfb + 400,
            "barge_in_stop_ms": 75,
            "interruption_recovery_ms": 150,
            "stage_latency_ms": {
                "asr_finalization_ms": 40,
                "llm_ttft_ms": 110,
                "tts_first_chunk_ms": 50,
            },
        }
        return trace

    report = run_openvoicecs_realtime_load(
        bench,
        ReferenceRealtimeClient(agent),
        max_scenarios=2,
        trials=2,
        concurrency_levels=(1, 2),
        region="test-region",
        network="loopback",
        hardware_profile="ci-macos-arm64",
        pricing_snapshot_date="2026-06-11",
        model_metadata={"agent": "metered"},
    )

    assert report["benchmark"] == "OpenVoiceCS-Realtime-Load"
    assert report["pass_k"] == 1.0
    assert report["num_trials_per_scenario"] == 4
    assert report["reference_client"]["concurrency_levels"] == [1, 2]
    assert report["reference_client"]["hardware_profile"] == "ci-macos-arm64"
    assert report["operational_metrics"]["load"]["1"]["count"] == 4
    assert report["operational_metrics"]["load"]["2"]["count"] == 4
    assert report["operational_metrics"]["v2v_ttfb_ms"]["p95"] == 202.0
    assert report["operational_metrics"]["barge_in_stop_ms"]["p95"] == 75.0
    assert report["operational_metrics"]["interruption_recovery_ms"]["p95"] == 150.0

    frontier = build_frontier_report([report])
    load = frontier["scorecards"]["metered"]["latency_load"]["levels"]
    assert load["1"]["sample_count"] == 4
    assert load["1"]["saturated"] is True
    assert load["2"]["sample_count"] == 4
    assert load["2"]["saturated"] is True
    assert load["2"]["p95_v2v_ttfb_ms"] == 202.0


def test_realtime_load_harness_records_failed_calls_without_aborting():
    bench = OpenVoiceCSBench.load()

    def agent(request):
        if request["concurrency"] == 2 and request["trial_index"] == 1:
            raise RuntimeError("capacity exceeded")
        trace = oracle_agent(request["scenario"], request["trial_index"])
        ttfb = 200 + request["concurrency"]
        trace["cost_usd"] = 0.02
        trace["latency"] = {
            "v2v_ttfb_ms": ttfb,
            "v2v_last_byte_ms": ttfb + 400,
            "stage_latency_ms": {
                "asr_finalization_ms": 40,
                "llm_ttft_ms": 110,
                "tts_first_chunk_ms": 50,
            },
        }
        return trace

    report = run_openvoicecs_realtime_load(
        bench,
        ReferenceRealtimeClient(agent),
        max_scenarios=1,
        trials=2,
        concurrency_levels=(1, 2),
        region="test-region",
        network="loopback",
        pricing_snapshot_date="2026-06-11",
        model_metadata={"agent": "capacity-limited"},
    )

    assert validate_report(report) == []
    load_2 = report["operational_metrics"]["load"]["2"]
    assert load_2["requested_calls"] == 2
    assert load_2["completed_calls"] == 1
    assert load_2["error_calls"] == 1
    assert load_2["error_rate"] == 0.5
    assert load_2["count"] == 1
    assert report["mean_pass_rate"] == 0.75
    failed = [
        trial for trial in report["results"][0]["trials"]
        if trial.get("error") == "capacity exceeded"
    ]
    assert len(failed) == 1
    assert failed[0]["passed"] is False
    assert failed[0]["scores"]["task_success"] == 0.0
    assert failed[0]["concurrency"] == 2

    frontier = build_frontier_report([report])
    scorecard_load_2 = frontier["scorecards"]["capacity-limited"]["latency_load"]["levels"]["2"]
    assert scorecard_load_2["requested_calls"] == 2
    assert scorecard_load_2["completed_calls"] == 1
    assert scorecard_load_2["error_calls"] == 1


def test_realtime_load_report_feeds_frontier():
    bench = OpenVoiceCSBench.load()

    def agent(request):
        trace = oracle_agent(request["scenario"], request["trial_index"])
        trace["cost_usd"] = 0.01
        trace["latency"] = {
            "v2v_ttfb_ms": 250,
            "v2v_last_byte_ms": 650,
            "stage_latency_ms": {
                "asr_finalization_ms": 50,
                "llm_ttft_ms": 140,
                "tts_first_chunk_ms": 60,
            },
        }
        return trace

    load_report = run_openvoicecs_realtime_load(
        bench,
        ReferenceRealtimeClient(agent),
        max_scenarios=1,
        trials=1,
        concurrency_levels=(1,),
        pricing_snapshot_date="2026-06-11",
        model_metadata={"display_name": "realtime-oracle"},
    )
    frontier = build_frontier_report([load_report])
    scorecard = frontier["scorecards"]["realtime-oracle"]

    assert frontier["frontier"] == ["realtime-oracle"]
    assert scorecard["p95_v2v_ttfb_ms"] == 250.0
    assert scorecard["cost_usd_per_successful_conversation"] == 0.01
    assert scorecard["latency_at_100_concurrency_p95_ms"] is None


def test_builtin_realtime_agent_emits_reference_cost_usage_for_frontier():
    bench = OpenVoiceCSBench.load()
    load_report = run_openvoicecs_realtime_load(
        bench,
        ReferenceRealtimeClient(builtin_realtime_agent("oracle")),
        max_scenarios=2,
        trials=1,
        concurrency_levels=(1,),
        region="local",
        network="loopback",
        pricing_snapshot_date="2026-06-11",
        model_metadata={
            "display_name": "builtin-realtime-oracle",
            "pricing_profile_id": "reference-zero-v0.1",
        },
    )

    frontier = build_frontier_report([load_report], pricing_manifest=load_pricing_manifest())
    scorecard = frontier["scorecards"]["builtin-realtime-oracle"]

    assert frontier["frontier"] == ["builtin-realtime-oracle"]
    assert scorecard["cost_usd_per_successful_conversation"] == 0.0
    assert scorecard["latency_measurement"]["event_stream_samples"] == 2
    assert scorecard["latency_measurement"]["vad_origin_samples"] == 2
    assert scorecard["latency_measurement"]["first_audio_event_samples"] == 2
    assert scorecard["latency_measurement"]["last_audio_event_samples"] == 2
    assert scorecard["cost_provenance"]["fully_loaded_samples"] == 2
    assert scorecard["cost_provenance"]["component_sample_counts"]["asr"] == 2
    assert scorecard["cost_provenance"]["component_sample_counts"]["transport"] == 2
