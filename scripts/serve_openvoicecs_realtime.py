#!/usr/bin/env python3
"""Serve a deterministic OpenVoiceCS realtime adapter over WebSocket."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.benchmark.realtime import builtin_realtime_agent


def _events_from_trace(trace: Any) -> list[dict[str, Any]]:
    if not isinstance(trace, dict):
        return [
            {
                "type": "agent.complete",
                "t_ms": 0.0,
                "messages": [{"role": "agent", "text": str(trace or "")}],
            }
        ]
    events = trace.get("events")
    if isinstance(events, list) and all(isinstance(event, dict) for event in events):
        return events
    return [
        {
            "type": "agent.complete",
            "t_ms": trace.get("latency_ms", 0.0),
            "messages": trace.get("messages") or [],
            "tool_calls": trace.get("tool_calls") or [],
            "events": trace.get("events") or [],
            "usage": trace.get("usage") or {},
            "cost_usd": trace.get("cost_usd"),
        }
    ]


async def _serve(args: argparse.Namespace) -> None:
    try:
        import websockets
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("websockets is required to serve the realtime endpoint") from exc

    agent = builtin_realtime_agent(args.agent)
    delay_seconds = args.event_delay_ms / 1000.0

    async def handler(websocket: Any) -> None:
        raw = await websocket.recv()
        request = json.loads(raw)
        trace = agent(request)
        if inspect.isawaitable(trace):
            trace = await trace
        for event in _events_from_trace(trace):
            await websocket.send(json.dumps(event))
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

    async with websockets.serve(handler, args.host, args.port) as server:
        bound_port = args.port
        if server.sockets:
            bound_port = server.sockets[0].getsockname()[1]
        print(
            f"Serving OpenVoiceCS {args.agent} realtime endpoint "
            f"at ws://{args.host}:{bound_port}",
            flush=True,
        )
        await asyncio.Future()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=["oracle", "noop"], default="oracle")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--event-delay-ms",
        type=float,
        default=0.0,
        help="Optional delay between emitted realtime events.",
    )
    args = parser.parse_args()
    asyncio.run(_serve(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
