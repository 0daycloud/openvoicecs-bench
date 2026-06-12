#!/usr/bin/env python3
"""Generate deterministic realtime oracle reports for OpenVoiceCS release bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.benchmark.openvoicecs import (
    OpenVoiceCSBench,
    build_audio_variant_scenarios,
    load_audio_manifest,
)
from src.evaluation.benchmark.realtime import (
    ReferenceRealtimeClient,
    WebSocketRealtimeClient,
    builtin_realtime_agent,
    run_openvoicecs_realtime_load,
)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _metadata(
    display_name: str,
    *,
    input_modality: str,
    transport: str,
    endpoint: str | None,
) -> dict[str, Any]:
    return {
        "display_name": display_name,
        "agent": "oracle",
        "provider": "reference",
        "model_id": f"{display_name}-realtime-v0.1",
        "pricing_profile_id": "reference-zero-v0.1",
        "pricing_snapshot_date": "2026-06-11",
        "pipeline_type": "cascaded",
        "input_modality": input_modality,
        "transport": transport,
        "endpoint": endpoint,
        "baseline_id": display_name,
    }


def _run_report(
    bench: OpenVoiceCSBench,
    *,
    display_name: str,
    input_modality: str,
    trials: int,
    concurrency_levels: tuple[int, ...],
    region: str,
    network: str,
    hardware_profile: str,
    seed: int,
    transport: str,
    endpoint: str | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    if transport == "websocket":
        if not endpoint:
            raise ValueError("endpoint is required for websocket transport")
        client = WebSocketRealtimeClient(endpoint, timeout_seconds=timeout_seconds)
    elif transport == "in_process":
        client = ReferenceRealtimeClient(builtin_realtime_agent("oracle"))
    else:
        raise ValueError(f"unsupported transport: {transport}")
    return run_openvoicecs_realtime_load(
        bench,
        client,
        trials=trials,
        concurrency_levels=concurrency_levels,
        region=region,
        network=network,
        hardware_profile=hardware_profile,
        seed=seed,
        pricing_snapshot_date="2026-06-11",
        model_metadata=_metadata(
            display_name,
            input_modality=input_modality,
            transport=transport,
            endpoint=endpoint,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default="data/openvoicecs/scenarios_v0.1.json")
    parser.add_argument("--audio-manifest", default="data/openvoicecs/audio_manifest_v0.1.json")
    parser.add_argument("--output-dir", default="data/openvoicecs/reports")
    parser.add_argument("--region", default="local")
    parser.add_argument("--network", default="loopback")
    parser.add_argument("--hardware-profile", default="local-macos-arm64")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency-levels", nargs="+", type=int, default=[1, 10, 100])
    parser.add_argument("--text-trials", type=int, default=2)
    parser.add_argument("--audio-trials", type=int, default=4)
    parser.add_argument("--transport", choices=["in_process", "websocket"], default="in_process")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()

    base_bench = OpenVoiceCSBench.load(args.scenarios)
    concurrency_levels = tuple(args.concurrency_levels)
    output_dir = Path(args.output_dir)

    text_report = _run_report(
        base_bench,
        display_name="oracle_text",
        input_modality="text",
        trials=args.text_trials,
        concurrency_levels=concurrency_levels,
        region=args.region,
        network=args.network,
        hardware_profile=args.hardware_profile,
        seed=args.seed,
        transport=args.transport,
        endpoint=args.endpoint,
        timeout_seconds=args.timeout_seconds,
    )
    text_path = output_dir / "oracle_text_realtime.json"
    _write_json(text_path, text_report)

    audio_variants = load_audio_manifest(args.audio_manifest)
    audio_scenarios = build_audio_variant_scenarios(base_bench.scenarios, audio_variants)
    audio_bench = OpenVoiceCSBench(
        scenarios=audio_scenarios,
        metadata={
            **base_bench.metadata,
            "source_scenario_count": len(base_bench.scenarios),
            "audio_manifest_path": args.audio_manifest,
            "evaluation_mode": "audio_manifest_realtime",
        },
        version=base_bench.version,
    )
    audio_report = _run_report(
        audio_bench,
        display_name="oracle_audio_manifest",
        input_modality="audio",
        trials=args.audio_trials,
        concurrency_levels=concurrency_levels,
        region=args.region,
        network=args.network,
        hardware_profile=args.hardware_profile,
        seed=args.seed,
        transport=args.transport,
        endpoint=args.endpoint,
        timeout_seconds=args.timeout_seconds,
    )
    audio_report["evaluation_mode"] = "audio_manifest_realtime"
    audio_report["audio_manifest_path"] = args.audio_manifest
    audio_report["num_audio_variants"] = len(audio_scenarios)
    audio_path = output_dir / "oracle_audio_manifest_realtime.json"
    _write_json(audio_path, audio_report)

    print(f"Saved text realtime report:  {text_path}")
    print(f"Saved audio realtime report: {audio_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
