#!/usr/bin/env python3
"""Generate synthetic WAV assets for OpenVoiceCS audio-manifest variants.

    The script uses the local macOS ``say`` command for deterministic synthetic
    speech, writes 16 kHz mono PCM WAV assets, and optionally applies
    deterministic background noise for robustness variants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import struct
import subprocess
import tempfile
import wave
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-manifest",
        default="data/openvoicecs/audio_manifest_v0.1.json",
        help="OpenVoiceCS audio manifest JSON path",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Optional macOS say voice name; default uses the system voice",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Skip variants whose WAV path already exists",
    )
    args = parser.parse_args()

    if shutil.which("say") is None:
        raise SystemExit("macOS 'say' command is required")

    manifest_path = Path(args.audio_manifest)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    variants = manifest.get("variants")
    if not isinstance(variants, list):
        raise SystemExit("audio manifest must contain a variants list")

    generated = []
    with tempfile.TemporaryDirectory(prefix="openvoicecs-audio-") as tmp:
        temp_dir = Path(tmp)
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            variant_id = str(variant.get("id"))
            transcript = str(variant.get("transcript", "")).strip()
            audio = variant.get("audio", {})
            if not transcript or not isinstance(audio, dict) or not audio.get("path"):
                raise SystemExit(f"{variant_id}: missing transcript or audio.path")
            output = Path(audio["path"])
            if args.only_missing and output.exists():
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            wav_path = temp_dir / f"{_safe_name(variant_id)}.wav"

            say_cmd = [
                "say",
                "--file-format=WAVE",
                "--data-format=LEI16@16000",
                "-o",
                str(wav_path),
            ]
            if args.voice:
                say_cmd.extend(["-v", args.voice])
            say_cmd.append(transcript)
            subprocess.run(say_cmd, check=True)
            _validate_nonempty_wav(wav_path, variant_id)
            if _has_background_noise(variant):
                _apply_background_noise(wav_path, variant_id, _snr_db(variant))
            shutil.copy2(wav_path, output)
            generated.append(str(output))

    print(f"Generated {len(generated)} audio asset(s)")
    for path in generated:
        print(path)
    return 0


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _has_background_noise(variant: dict[str, Any]) -> bool:
    return any(
        isinstance(perturbation, dict) and perturbation.get("type") == "background_noise"
        for perturbation in variant.get("perturbations", [])
    )


def _snr_db(variant: dict[str, Any]) -> float:
    for perturbation in variant.get("perturbations", []):
        if isinstance(perturbation, dict) and perturbation.get("type") == "background_noise":
            value = perturbation.get("snr_db")
            if isinstance(value, (int, float)):
                return float(value)
    return 12.0


def _apply_background_noise(path: Path, variant_id: str, snr_db: float) -> None:
    with wave.open(str(path), "rb") as reader:
        params = reader.getparams()
        frames = reader.readframes(reader.getnframes())
    if params.sampwidth != 2 or params.nchannels != 1:
        raise SystemExit(f"{variant_id}: expected 16-bit mono WAV after conversion")

    samples = list(struct.unpack(f"<{len(frames) // 2}h", frames))
    if not samples:
        return
    rms = (sum(sample * sample for sample in samples) / len(samples)) ** 0.5
    noise_rms = rms / (10 ** (snr_db / 20.0)) if rms else 250.0
    rng = random.Random(int(hashlib.sha256(variant_id.encode("utf-8")).hexdigest()[:16], 16))
    noisy = []
    for sample in samples:
        value = int(round(sample + rng.gauss(0.0, noise_rms)))
        noisy.append(max(-32768, min(32767, value)))

    with wave.open(str(path), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(struct.pack(f"<{len(noisy)}h", *noisy))


def _validate_nonempty_wav(path: Path, variant_id: str) -> None:
    with wave.open(str(path), "rb") as reader:
        params = reader.getparams()
        frames = reader.getnframes()
    if params.sampwidth != 2 or params.nchannels != 1 or params.framerate != 16000:
        raise SystemExit(f"{variant_id}: expected 16-bit mono 16000 Hz WAV")
    if frames <= 0:
        raise SystemExit(
            f"{variant_id}: generated zero-frame WAV; run outside sandbox or use another TTS"
        )


if __name__ == "__main__":
    raise SystemExit(main())
