"""Reference realtime client and load harness for voice-agent frontier runs.

This module drives a candidate system through one canonical conversation-level
request contract, records voice-to-voice timings, and feeds the resulting trace
through OpenVoiceCS scoring. The same harness can call an in-process adapter for
tests or a WebSocket endpoint for deployed systems.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
import inspect
import json
import statistics
import time
from typing import Any, Callable
from urllib import request as urllib_request

from src.evaluation.benchmark.openvoicecs import (
    METRIC_NAMES,
    OpenVoiceCSBench,
    no_op_agent,
    oracle_agent,
)

RealtimeAgentFn = Callable[[dict[str, Any]], Any]


DEFAULT_CONCURRENCY_LEVELS = (1, 10, 100)
REALTIME_BENCH_VERSION = "0.1.0"


@dataclass(frozen=True)
class ReferenceRealtimeConfig:
    """Frozen metadata for a realtime benchmark release/run."""

    region: str = "unspecified"
    network: str = "unspecified"
    hardware_profile: str = "unspecified"
    transport: str = "in_process"
    concurrency_levels: tuple[int, ...] = DEFAULT_CONCURRENCY_LEVELS
    seed: int = 0
    pricing_snapshot_date: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "network": self.network,
            "hardware_profile": self.hardware_profile,
            "transport": self.transport,
            "concurrency_levels": list(self.concurrency_levels),
            "seed": self.seed,
            "pricing_snapshot_date": self.pricing_snapshot_date,
        }


class ReferenceRealtimeClient:
    """Canonical realtime client for in-process voice-agent adapters."""

    def __init__(self, agent_fn: RealtimeAgentFn) -> None:
        self.agent_fn = agent_fn
        self.transport = "in_process"

    async def run_call(
        self,
        scenario: dict[str, Any],
        *,
        trial_index: int,
        concurrency: int,
        seed: int,
    ) -> dict[str, Any]:
        request = build_realtime_request(
            scenario,
            trial_index=trial_index,
            concurrency=concurrency,
            seed=seed,
            transport=self.transport,
        )
        started = time.perf_counter()
        await asyncio.sleep(0)
        raw = self.agent_fn(request)
        if inspect.isawaitable(raw):
            raw = await raw
        measured_ms = (time.perf_counter() - started) * 1000
        if _has_canonical_realtime_events(raw):
            return normalize_realtime_events(raw["events"], measured_ms=measured_ms)
        return normalize_realtime_trace(raw, measured_ms=measured_ms)


class WebSocketRealtimeClient:
    """Reference WebSocket client using the same request contract.

    The endpoint is expected to accept one JSON request and emit JSON events
    until an ``agent.complete`` or ``error`` event. Server-provided latency fields
    are respected; otherwise client receive times are used.
    """

    def __init__(self, endpoint: str, *, timeout_seconds: float = 30.0) -> None:
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.transport = "websocket"

    async def run_call(
        self,
        scenario: dict[str, Any],
        *,
        trial_index: int,
        concurrency: int,
        seed: int,
    ) -> dict[str, Any]:
        try:
            import websockets
        except ImportError as exc:  # pragma: no cover - depends on optional env package
            raise RuntimeError("websockets is required for WebSocket realtime runs") from exc

        request = build_realtime_request(
            scenario,
            trial_index=trial_index,
            concurrency=concurrency,
            seed=seed,
            transport=self.transport,
        )
        started = time.perf_counter()
        events: list[dict[str, Any]] = []
        async with websockets.connect(self.endpoint, open_timeout=self.timeout_seconds) as websocket:
            await websocket.send(json.dumps(request))
            while True:
                raw_message = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=self.timeout_seconds,
                )
                received_ms = (time.perf_counter() - started) * 1000
                event = json.loads(raw_message)
                if isinstance(event, dict):
                    event.setdefault("client_received_ms", received_ms)
                    events.append(event)
                    if event.get("type") in {"agent.complete", "error"}:
                        break
        measured_ms = (time.perf_counter() - started) * 1000
        return normalize_realtime_events(events, measured_ms=measured_ms)


class WebRTCRealtimeClient:
    """Reference WebRTC client using a JSON data-channel event stream.

    The signaling endpoint receives a JSON offer ``{"sdp": ..., "type": "offer"}``
    and returns a JSON answer. Once the data channel opens, the client sends the
    canonical OpenVoiceCS realtime request and consumes JSON events until
    ``agent.complete`` or ``error``.
    """

    def __init__(
        self,
        signaling_endpoint: str,
        *,
        timeout_seconds: float = 30.0,
        data_channel: str = "openvoicecs",
    ) -> None:
        self.signaling_endpoint = signaling_endpoint
        self.timeout_seconds = timeout_seconds
        self.data_channel = data_channel
        self.transport = "webrtc"

    async def run_call(
        self,
        scenario: dict[str, Any],
        *,
        trial_index: int,
        concurrency: int,
        seed: int,
    ) -> dict[str, Any]:
        try:
            from aiortc import RTCSessionDescription
            from aiortc import RTCPeerConnection
        except ImportError as exc:  # pragma: no cover - depends on optional env package
            raise RuntimeError("aiortc is required for WebRTC realtime runs") from exc

        request = build_realtime_request(
            scenario,
            trial_index=trial_index,
            concurrency=concurrency,
            seed=seed,
            transport=self.transport,
        )
        started = time.perf_counter()
        peer = RTCPeerConnection()
        channel = peer.createDataChannel(self.data_channel)
        opened = asyncio.Event()
        completed = asyncio.Event()
        events: list[dict[str, Any]] = []

        @channel.on("open")
        def _on_open() -> None:
            opened.set()

        @channel.on("message")
        def _on_message(message: Any) -> None:
            received_ms = (time.perf_counter() - started) * 1000
            if isinstance(message, bytes):
                message = message.decode("utf-8")
            try:
                event = json.loads(str(message))
            except json.JSONDecodeError:
                event = {"type": "error", "message": "invalid JSON data-channel message"}
            if isinstance(event, dict):
                event.setdefault("client_received_ms", received_ms)
                events.append(event)
                if event.get("type") in {"agent.complete", "error"}:
                    completed.set()

        try:
            offer = await peer.createOffer()
            await peer.setLocalDescription(offer)
            answer = await asyncio.to_thread(
                _post_webrtc_offer,
                self.signaling_endpoint,
                peer.localDescription.sdp,
                self.timeout_seconds,
            )
            await peer.setRemoteDescription(
                RTCSessionDescription(sdp=answer["sdp"], type=answer.get("type", "answer"))
            )
            await asyncio.wait_for(opened.wait(), timeout=self.timeout_seconds)
            channel.send(json.dumps(request))
            await asyncio.wait_for(completed.wait(), timeout=self.timeout_seconds)
        finally:
            await peer.close()

        measured_ms = (time.perf_counter() - started) * 1000
        return normalize_realtime_events(events, measured_ms=measured_ms)


def run_openvoicecs_realtime_load(
    bench: OpenVoiceCSBench,
    client: ReferenceRealtimeClient | WebSocketRealtimeClient | WebRTCRealtimeClient,
    *,
    max_scenarios: int | None = None,
    trials: int = 1,
    track: str | None = None,
    concurrency_levels: tuple[int, ...] | list[int] = DEFAULT_CONCURRENCY_LEVELS,
    region: str = "unspecified",
    network: str = "unspecified",
    hardware_profile: str = "unspecified",
    seed: int = 0,
    pricing_snapshot_date: str | None = None,
    model_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run OpenVoiceCS scenarios under controlled realtime concurrency levels."""
    return asyncio.run(
        run_openvoicecs_realtime_load_async(
            bench,
            client,
            max_scenarios=max_scenarios,
            trials=trials,
            track=track,
            concurrency_levels=tuple(concurrency_levels),
            region=region,
            network=network,
            hardware_profile=hardware_profile,
            seed=seed,
            pricing_snapshot_date=pricing_snapshot_date,
            model_metadata=model_metadata,
        )
    )


async def run_openvoicecs_realtime_load_async(
    bench: OpenVoiceCSBench,
    client: ReferenceRealtimeClient | WebSocketRealtimeClient | WebRTCRealtimeClient,
    *,
    max_scenarios: int | None = None,
    trials: int = 1,
    track: str | None = None,
    concurrency_levels: tuple[int, ...] = DEFAULT_CONCURRENCY_LEVELS,
    region: str = "unspecified",
    network: str = "unspecified",
    hardware_profile: str = "unspecified",
    seed: int = 0,
    pricing_snapshot_date: str | None = None,
    model_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if not concurrency_levels or any(level < 1 for level in concurrency_levels):
        raise ValueError("concurrency_levels must contain positive integers")

    scenarios = bench.scenarios
    if track:
        scenarios = [scenario for scenario in scenarios if scenario["track"] == track]
    if max_scenarios:
        scenarios = scenarios[:max_scenarios]

    config = ReferenceRealtimeConfig(
        region=region,
        network=network,
        hardware_profile=hardware_profile,
        transport=client.transport,
        concurrency_levels=tuple(concurrency_levels),
        seed=seed,
        pricing_snapshot_date=pricing_snapshot_date,
    )
    started = time.perf_counter()
    scenario_trial_results: dict[str, list[dict[str, Any]]] = {
        scenario["id"]: [] for scenario in scenarios
    }
    load_runs: dict[str, dict[str, Any]] = {}

    for concurrency in concurrency_levels:
        requests = [
            (scenario, trial_index)
            for scenario in scenarios
            for trial_index in range(trials)
        ]
        semaphore = asyncio.Semaphore(concurrency)
        active_calls = 0
        peak_active_calls = 0
        active_lock = asyncio.Lock()
        level_started = time.perf_counter()

        async def run_one(scenario: dict[str, Any], trial_index: int) -> tuple[str, dict[str, Any]]:
            nonlocal active_calls, peak_active_calls
            async with semaphore:
                async with active_lock:
                    active_calls += 1
                    peak_active_calls = max(peak_active_calls, active_calls)
                try:
                    trace = await client.run_call(
                        deepcopy(scenario),
                        trial_index=trial_index,
                        concurrency=concurrency,
                        seed=seed,
                    )
                    scored = bench._score_single_trial(
                        scenario=deepcopy(scenario),
                        agent_fn=lambda _scenario, _trial: trace,
                        trial_index=trial_index,
                    )
                except Exception as exc:
                    scored = _failed_realtime_trial(trial_index=trial_index, error=exc)
                finally:
                    async with active_lock:
                        active_calls -= 1
                scored["concurrency"] = concurrency
                scored["transport"] = client.transport
                return scenario["id"], scored

        level_results = await asyncio.gather(
            *(run_one(scenario, trial_index) for scenario, trial_index in requests)
        )
        completed_calls = sum(1 for _scenario_id, scored in level_results if "error" not in scored)
        error_calls = len(level_results) - completed_calls
        load_runs[str(concurrency)] = {
            "target_concurrency": concurrency,
            "requested_calls": len(requests),
            "completed_calls": completed_calls,
            "error_calls": error_calls,
            "error_rate": round(error_calls / len(requests), 6) if requests else 0.0,
            "peak_active_calls": peak_active_calls,
            "saturated": peak_active_calls >= concurrency if requests else False,
            "wall_seconds": round(time.perf_counter() - level_started, 3),
        }

        for scenario_id, scored in level_results:
            scenario_trial_results[scenario_id].append(scored)

    results = [
        bench._aggregate_scenario_trials(scenario, scenario_trial_results[scenario["id"]])
        for scenario in scenarios
    ]
    report = bench._aggregate_results(
        results,
        trials=trials * len(concurrency_levels),
    )
    report["benchmark"] = "OpenVoiceCS-Realtime-Load"
    report["benchmark_version"] = REALTIME_BENCH_VERSION
    report["source_benchmark"] = "OpenVoiceCS-Bench"
    report["source_benchmark_version"] = bench.version
    report["model_metadata"] = {
        **(model_metadata or {}),
        "pricing_snapshot_date": pricing_snapshot_date,
    }
    report["reference_client"] = config.to_dict()
    report["environment"] = config.to_dict()
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    report["operational_metrics"] = {
        **report.get("operational_metrics", {}),
        **_aggregate_realtime_operational_metrics(
            results,
            concurrency_levels,
            load_runs=load_runs,
        ),
    }
    return report


def build_realtime_request(
    scenario: dict[str, Any],
    *,
    trial_index: int,
    concurrency: int,
    seed: int,
    transport: str,
) -> dict[str, Any]:
    """Build the canonical request sent to every realtime system."""
    audio_variant = scenario.get("audio_variant") or {}
    audio = audio_variant.get("audio") if isinstance(audio_variant, dict) else {}
    transcript = (
        audio_variant.get("transcript")
        if isinstance(audio_variant, dict)
        else None
    ) or _scenario_transcript(scenario)
    return {
        "protocol": "openvoicecs.realtime.v1",
        "transport": transport,
        "scenario_id": scenario["id"],
        "scenario": deepcopy(scenario),
        "base_scenario_id": scenario.get("base_scenario_id"),
        "trial_index": trial_index,
        "concurrency": concurrency,
        "seed": seed,
        "domain": scenario.get("domain"),
        "track": scenario.get("track"),
        "input_modality": scenario.get("input_modality", "audio"),
        "customer_goal": scenario.get("customer_goal"),
        "transcript": transcript,
        "audio": audio or None,
        "events": [
            {"type": "session.start", "t_ms": 0.0},
            {
                "type": "user.end_speech",
                "t_ms": 0.0,
                "definition": "VAD endpoint; latency origin for TTFB and last byte.",
            },
        ],
        "expected_outputs": {
            "first_audio_event": "tts.first_audio",
            "last_audio_event": "agent.complete",
            "interruption_event": "barge_in.stop",
        },
    }


def _post_webrtc_offer(
    endpoint: str,
    sdp: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    payload = json.dumps({"sdp": sdp, "type": "offer"}).encode("utf-8")
    request = urllib_request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
        answer = json.loads(response.read().decode("utf-8"))
    if not isinstance(answer, dict) or not answer.get("sdp"):
        raise RuntimeError("WebRTC signaling endpoint must return an SDP answer")
    return answer


def normalize_realtime_trace(raw_trace: Any, *, measured_ms: float) -> dict[str, Any]:
    """Normalize an in-process or server-complete trace for OpenVoiceCS scoring."""
    if raw_trace is None:
        raw_trace = {}
    if isinstance(raw_trace, str):
        raw_trace = {"messages": [{"role": "agent", "text": raw_trace}]}
    if not isinstance(raw_trace, dict):
        raise TypeError("realtime agent must return a string or dict trace")

    latency = _normalize_latency_dict(raw_trace.get("latency"), measured_ms)
    latency.setdefault(
        "measurement",
        {
            "source": "reported_latency" if raw_trace.get("latency") is not None else "runtime_fallback",
            "origin_event": None,
            "origin_t_ms": None,
        },
    )
    return {
        "messages": raw_trace.get("messages") or raw_trace.get("responses") or [],
        "tool_calls": raw_trace.get("tool_calls") or [],
        "events": raw_trace.get("events") or [],
        "claims": raw_trace.get("claims") or raw_trace.get("grounding_claims") or [],
        "usage": raw_trace.get("usage") or {},
        "cost_usd": raw_trace.get("cost_usd"),
        "latency": latency,
        "latency_ms": latency.get("v2v_ttfb_ms", measured_ms),
    }


def normalize_realtime_events(
    events: list[dict[str, Any]],
    *,
    measured_ms: float,
) -> dict[str, Any]:
    """Normalize a stream of WebSocket events into an OpenVoiceCS trace."""
    messages = []
    tool_calls = []
    policy_events = []
    usage: dict[str, Any] = {}
    cost_usd = None
    latency: dict[str, Any] = {}
    seen_event_types = set()

    for event in events:
        event_type = event.get("type")
        if event_type:
            seen_event_types.add(str(event_type))
        if event_type == "agent.message":
            messages.append({"role": "agent", "text": str(event.get("text", ""))})
        elif event_type == "tool.call":
            tool_calls.append({
                "name": event.get("name"),
                "arguments": event.get("arguments") or {},
            })
        elif event_type == "policy.event":
            if event.get("name"):
                policy_events.append(str(event["name"]))
        elif event_type == "usage":
            usage.update(event.get("usage") or {})
        elif event_type == "cost":
            cost_usd = event.get("cost_usd")
        elif event_type == "asr.final":
            latency["asr_finalization_ms"] = event.get("t_ms", event.get("client_received_ms"))
        elif event_type == "llm.first_token":
            latency["llm_ttft_ms"] = event.get("t_ms", event.get("client_received_ms"))
        elif event_type == "tts.first_audio":
            latency["v2v_ttfb_ms"] = event.get("t_ms", event.get("client_received_ms"))
            latency["tts_first_chunk_ms"] = event.get("tts_first_chunk_ms")
        elif event_type == "barge_in.stop":
            latency["barge_in_stop_ms"] = event.get("t_ms", event.get("client_received_ms"))
        elif event_type == "barge_in.recovered":
            latency["interruption_recovery_ms"] = event.get("t_ms", event.get("client_received_ms"))
        elif event_type == "agent.complete":
            latency["v2v_last_byte_ms"] = event.get("t_ms", event.get("client_received_ms"))
            if event.get("messages"):
                messages.extend(event["messages"])
            if event.get("tool_calls"):
                tool_calls.extend(event["tool_calls"])
            if event.get("events"):
                policy_events.extend(event["events"])
            if event.get("usage"):
                usage.update(event["usage"])
            if event.get("cost_usd") is not None:
                cost_usd = event["cost_usd"]

    latency["measurement"] = {
        "source": "event_stream",
        "origin_event": "user.end_speech",
        "origin_t_ms": 0.0,
        "first_audio_event": "tts.first_audio" if "tts.first_audio" in seen_event_types else None,
        "last_audio_event": "agent.complete" if "agent.complete" in seen_event_types else None,
        "barge_in_stop_event": "barge_in.stop" if "barge_in.stop" in seen_event_types else None,
        "interruption_recovery_event": (
            "barge_in.recovered" if "barge_in.recovered" in seen_event_types else None
        ),
    }

    return normalize_realtime_trace(
        {
            "messages": messages,
            "tool_calls": tool_calls,
            "events": policy_events,
            "usage": usage,
            "cost_usd": cost_usd,
            "latency": latency,
        },
        measured_ms=measured_ms,
    )


def builtin_realtime_agent(name: str) -> RealtimeAgentFn:
    """Return a deterministic in-process realtime adapter for demos/tests."""
    if name == "oracle":
        base_agent = oracle_agent
    elif name == "noop":
        base_agent = no_op_agent
    else:
        raise ValueError(f"unknown built-in realtime agent: {name}")

    def agent(request: dict[str, Any]) -> dict[str, Any]:
        scenario = request.get("scenario") or request
        trace = base_agent(scenario, request.get("trial_index", 0))
        latency_ms = trace.get("latency_ms", 750)
        last_byte_ms = latency_ms + 350
        asr_ms = 50
        llm_ms = max(0, latency_ms - 120)
        tts_ms = 70
        usage = dict(trace.get("usage") or {})
        usage.update({
            "asr_seconds": usage.get("asr_seconds", 60),
            "input_tokens": usage.get("input_tokens", 1000),
            "output_tokens": usage.get("output_tokens", 250),
            "tts_characters": usage.get("tts_characters", 800),
            "call_duration_seconds": usage.get("call_duration_seconds", 60),
            "transport_seconds": usage.get("transport_seconds", 60),
        })
        trace["latency"] = {
            "v2v_ttfb_ms": latency_ms,
            "v2v_last_byte_ms": last_byte_ms,
            "stage_latency_ms": {
                "asr_finalization_ms": asr_ms,
                "llm_ttft_ms": llm_ms,
                "tts_first_chunk_ms": tts_ms,
            },
        }
        trace["usage"] = usage
        trace.setdefault("cost_usd", 0.0)
        trace["events"] = [
            {"type": "asr.final", "t_ms": asr_ms},
            {"type": "llm.first_token", "t_ms": llm_ms},
            {
                "type": "tts.first_audio",
                "t_ms": latency_ms,
                "tts_first_chunk_ms": tts_ms,
            },
            {"type": "barge_in.stop", "t_ms": min(120, latency_ms)},
            {"type": "barge_in.recovered", "t_ms": min(240, last_byte_ms)},
            {
                "type": "agent.complete",
                "t_ms": last_byte_ms,
                "messages": trace.get("messages") or [],
                "tool_calls": trace.get("tool_calls") or [],
                "events": trace.get("events") or [],
                "usage": usage,
                "cost_usd": trace.get("cost_usd"),
            },
        ]
        return trace

    return agent


def _has_canonical_realtime_events(raw_trace: Any) -> bool:
    if not isinstance(raw_trace, dict):
        return False
    events = raw_trace.get("events")
    if not isinstance(events, list):
        return False
    event_types = {
        event.get("type")
        for event in events
        if isinstance(event, dict) and isinstance(event.get("type"), str)
    }
    return "tts.first_audio" in event_types and "agent.complete" in event_types


def _scenario_transcript(scenario: dict[str, Any]) -> str:
    turns = scenario.get("conversation") or []
    if not turns:
        return scenario.get("customer_goal", "")
    return "\n".join(
        f"{turn.get('role', 'customer')}: {turn.get('text', '')}"
        for turn in turns
    )


def _normalize_latency_dict(raw_latency: Any, measured_ms: float) -> dict[str, Any]:
    if isinstance(raw_latency, (int, float)):
        raw_latency = {"v2v_ttfb_ms": float(raw_latency)}
    latency = dict(raw_latency) if isinstance(raw_latency, dict) else {}
    ttfb = _first_number(
        latency.get("v2v_ttfb_ms"),
        latency.get("voice_to_voice_ttfb_ms"),
        latency.get("ttfb_ms"),
        measured_ms,
    )
    last_byte = _first_number(
        latency.get("v2v_last_byte_ms"),
        latency.get("voice_to_voice_last_byte_ms"),
        latency.get("last_byte_ms"),
        measured_ms,
    )
    normalized = {
        "v2v_ttfb_ms": round(ttfb, 3),
        "v2v_last_byte_ms": round(max(last_byte, ttfb), 3),
    }
    stage = latency.get("stage_latency_ms") or latency.get("pipeline_ms") or {}
    if not isinstance(stage, dict):
        stage = {}
    stage_aliases = {
        "asr_finalization_ms": ("asr_finalization_ms", "asr_ms"),
        "llm_ttft_ms": ("llm_ttft_ms", "llm_ms"),
        "tts_first_chunk_ms": ("tts_first_chunk_ms", "tts_ms"),
    }
    normalized["stage_latency_ms"] = {
        canonical: _first_number(*(stage.get(name) for name in names), latency.get(canonical))
        for canonical, names in stage_aliases.items()
    }
    interruption = _first_number(
        latency.get("interruption_recovery_ms"),
        latency.get("barge_in_recovery_ms"),
    )
    stop = _first_number(
        latency.get("barge_in_stop_ms"),
        latency.get("interruption_stop_ms"),
        latency.get("stop_speaking_ms"),
    )
    if stop is not None:
        normalized["barge_in_stop_ms"] = round(stop, 3)
    if interruption is not None:
        normalized["interruption_recovery_ms"] = round(interruption, 3)
    measurement = latency.get("measurement") or latency.get("measurement_metadata")
    if isinstance(measurement, dict):
        normalized["measurement"] = measurement
    return normalized


def _aggregate_realtime_operational_metrics(
    results: list[dict[str, Any]],
    concurrency_levels: tuple[int, ...],
    *,
    load_runs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    by_concurrency: dict[int, list[dict[str, Any]]] = {
        level: [] for level in concurrency_levels
    }
    all_trials = []
    for scenario_result in results:
        for trial in scenario_result.get("trials", []):
            if "error" in trial:
                continue
            all_trials.append(trial)
            if trial.get("concurrency") in by_concurrency:
                by_concurrency[trial["concurrency"]].append(trial)

    load = {}
    for level, trials in by_concurrency.items():
        summary = _latency_summary(trials)
        summary.update((load_runs or {}).get(str(level), {}))
        load[str(level)] = summary
    all_summary = _latency_summary(all_trials)
    metrics = {
        "v2v_ttfb_ms": all_summary,
        "v2v_last_byte_ms": _latency_summary(all_trials, key="v2v_last_byte_ms"),
        "barge_in_stop_ms": _latency_summary(
            all_trials,
            key="barge_in_stop_ms",
        ),
        "interruption_recovery_ms": _latency_summary(
            all_trials,
            key="interruption_recovery_ms",
        ),
        "load": load,
    }
    if "100" in load:
        metrics["latency_at_100_concurrency_p95_ms"] = load["100"]["p95"]
    return metrics


def _failed_realtime_trial(*, trial_index: int, error: Exception) -> dict[str, Any]:
    return {
        "trial_index": trial_index,
        "error": str(error),
        "passed": False,
        "scores": {metric: 0.0 for metric in METRIC_NAMES},
    }


def _latency_summary(
    trials: list[dict[str, Any]],
    *,
    key: str = "v2v_ttfb_ms",
) -> dict[str, float | int | None]:
    values = [
        value for value in (
            _first_number(trial.get("latency", {}).get(key), trial.get(key))
            for trial in trials
        )
        if value is not None
    ]
    return {
        "count": len(values),
        "p50": _round_optional(statistics.median(values) if values else None, 3),
        "p90": _round_optional(_percentile(values, 90), 3),
        "p95": _round_optional(_percentile(values, 95), 3),
        "p99": _round_optional(_percentile(values, 99), 3),
    }


def _first_number(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _percentile(values: list[float | int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (percentile / 100)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _round_optional(value: float | None, ndigits: int) -> float | None:
    if value is None:
        return None
    return round(value, ndigits)
