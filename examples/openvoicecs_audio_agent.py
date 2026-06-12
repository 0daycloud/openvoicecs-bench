"""Minimal audio-variant adapter for OpenVoiceCS-Bench.

This example uses the manifest transcript as a stand-in for ASR output. A real
adapter would replace that step with STT over scenario["audio_variant"]["audio"]["path"].

Run:
    python examples/openvoicecs_audio_agent.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.benchmark import OpenVoiceCSBench, oracle_agent


def transcript_backed_audio_agent(scenario: dict, trial_index: int) -> dict:
    """Demo adapter for audio manifest variants.

    The transcript is deliberately only used to decide that the customer spoke;
    the tool trace comes from the oracle baseline so the example demonstrates
    the adapter shape, not a production agent.
    """
    audio_variant = scenario.get("audio_variant", {})
    transcript = audio_variant.get("transcript", "")
    trace = oracle_agent(scenario, trial_index)
    trace["asr"] = {
        "transcript": transcript,
        "audio_path": audio_variant.get("audio", {}).get("path"),
        "source": "manifest_transcript",
    }
    return trace


def main() -> None:
    bench = OpenVoiceCSBench.load()
    report = bench.score_audio_manifest(
        transcript_backed_audio_agent,
        track="robustness",
        trials=1,
        model_metadata={"agent": "transcript_backed_audio_demo"},
    )
    print(f"OpenVoiceCS audio score: {report['overall_score']:.2f} / 100")
    print(f"variants: {report['num_audio_variants']}")


if __name__ == "__main__":
    main()
