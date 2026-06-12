"""Tests for OpenVoiceCS leaderboard claim packages."""

from __future__ import annotations

from copy import deepcopy

from src.evaluation.benchmark.claims import (
    claims_stats,
    load_claims_manifest,
    validate_claims_manifest,
    validate_claims_manifest_file,
)


def test_claims_manifest_validates():
    manifest = load_claims_manifest()

    assert validate_claims_manifest(manifest) == []
    assert validate_claims_manifest_file() == []
    stats = claims_stats(manifest)
    assert stats["num_claims"] == 1
    assert stats["reference_fixture_claims"] == 1
    assert stats["official_claims"] == 0


def test_claims_manifest_detects_tampered_comparison_hash():
    manifest = load_claims_manifest()
    manifest["claims"][0]["comparison"]["sha256"] = "0" * 64

    messages = {
        (issue.path, issue.message)
        for issue in validate_claims_manifest(manifest)
    }

    assert ("claims[0].comparison.sha256", "does not match file contents") in messages


def test_claims_manifest_rejects_official_fixture_claim():
    manifest = load_claims_manifest()
    claim = deepcopy(manifest["claims"][0])
    claim["status"] = "official"
    claim["official_claim"] = True
    claim["release_bundle"] = {
        "path": "data/openvoicecs/releases/frontier_seed/release_bundle.json",
        "sha256": "8eb9505dd7e28b9a00ecd5d80caa926725fc66ce47085193cb0bd357179af56e",
        "bytes": 10794,
    }
    manifest["claims"] = [claim]

    messages = {
        (issue.path, issue.message)
        for issue in validate_claims_manifest(manifest)
    }

    assert (
        "claims[0].judging_evidence.annotation_mode",
        "official claims cannot use reference_fixture judging",
    ) in messages
