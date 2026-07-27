"""Tests for strict OpenVoiceCS audio asset validation."""

from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

from src.evaluation.benchmark.openvoicecs import (
    audio_asset_stats,
    pin_audio_manifest_assets_file,
    validate_audio_assets_file,
)


def _write_silence_wav(path: Path, *, sample_rate: int = 16000, seconds: float = 0.25) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frames)


def _manifest_for(path: str, sha256: str | None, duration_seconds: float = 0.25) -> dict:
    return {
        "name": "test audio manifest",
        "version": "0.1.0",
        "variants": [
            {
                "id": "retail-refund-damaged-item-001-clean-test",
                "scenario_id": "retail-refund-damaged-item-001",
                "track": "audio_to_action",
                "transcript": "My blender arrived cracked today.",
                "audio": {
                    "path": path,
                    "format": "wav",
                    "sample_rate_hz": 16000,
                    "duration_seconds": duration_seconds,
                    "sha256": sha256,
                },
                "perturbations": [],
            }
        ],
    }


def test_validate_audio_assets_accepts_matching_wav_metadata(tmp_path: Path):
    wav_path = tmp_path / "audio" / "clean.wav"
    _write_silence_wav(wav_path)
    sha256 = hashlib.sha256(wav_path.read_bytes()).hexdigest()
    manifest = _manifest_for("audio/clean.wav", sha256)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert validate_audio_assets_file(manifest_path, root_dir=tmp_path) == []

    stats = audio_asset_stats(manifest, root_dir=tmp_path)
    assert stats["num_existing_files"] == 1
    assert stats["num_sha256_verified"] == 1
    assert stats["num_sample_rate_verified"] == 1
    assert stats["num_duration_verified"] == 1


def test_validate_audio_assets_reports_missing_hash_file_and_bad_duration(tmp_path: Path):
    wav_path = tmp_path / "audio" / "clean.wav"
    _write_silence_wav(wav_path)
    manifest = _manifest_for("audio/clean.wav", None, duration_seconds=1.0)
    manifest["variants"].append(
        _manifest_for("audio/missing.wav", None)["variants"][0]
        | {"id": "retail-refund-damaged-item-001-missing-test"}
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    issues = validate_audio_assets_file(manifest_path, root_dir=tmp_path)
    messages = {(issue.scenario_id, issue.path, issue.message) for issue in issues}

    assert (
        "retail-refund-damaged-item-001-clean-test",
        "audio.sha256",
        "must be a SHA-256 hex digest",
    ) in messages
    assert (
        "retail-refund-damaged-item-001-clean-test",
        "audio.duration_seconds",
        "does not match wav duration",
    ) in messages
    assert (
        "retail-refund-damaged-item-001-missing-test",
        "audio.path",
        "file does not exist",
    ) in messages


def test_validate_audio_assets_rejects_zero_duration_placeholder(tmp_path: Path):
    wav_path = tmp_path / "audio" / "placeholder.wav"
    _write_silence_wav(wav_path, seconds=0.0)
    sha256 = hashlib.sha256(wav_path.read_bytes()).hexdigest()
    manifest = _manifest_for("audio/placeholder.wav", sha256, duration_seconds=0.0)
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "manifest_pinned.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    issues = validate_audio_assets_file(manifest_path, root_dir=tmp_path)
    messages = {(issue.scenario_id, issue.path, issue.message) for issue in issues}
    stats = audio_asset_stats(manifest, root_dir=tmp_path)
    pinned = pin_audio_manifest_assets_file(
        manifest_path,
        output_path=output_path,
        root_dir=tmp_path,
    )

    assert (
        "retail-refund-damaged-item-001-clean-test",
        "audio.duration_seconds",
        "must be positive",
    ) in messages
    assert stats["num_existing_files"] == 1
    assert stats["num_positive_duration_files"] == 0
    assert stats["num_duration_verified"] == 0
    assert pinned["output_path"] is None
    assert not output_path.exists()


def test_pin_audio_manifest_assets_file_writes_verified_metadata(tmp_path: Path):
    wav_path = tmp_path / "audio" / "clean.wav"
    _write_silence_wav(wav_path, sample_rate=8000, seconds=0.5)
    manifest = _manifest_for("audio/clean.wav", None, duration_seconds=999.0)
    manifest["variants"][0]["audio"] = {"path": "audio/clean.wav"}
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "manifest_pinned.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = pin_audio_manifest_assets_file(
        manifest_path,
        output_path=output_path,
        root_dir=tmp_path,
    )

    assert result["issues"] == []
    assert result["summary"]["num_pinned"] == 1
    pinned = json.loads(output_path.read_text(encoding="utf-8"))
    audio = pinned["variants"][0]["audio"]
    assert audio["format"] == "wav"
    assert audio["sample_rate_hz"] == 8000
    assert audio["duration_seconds"] == 0.5
    assert len(audio["sha256"]) == 64
    assert validate_audio_assets_file(output_path, root_dir=tmp_path) == []
