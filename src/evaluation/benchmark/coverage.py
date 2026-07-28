"""Coverage planning for OpenVoiceCS benchmark expansion."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.evaluation.benchmark.datapaths import data_path
from src.evaluation.benchmark.openvoicecs import (
    DEFAULT_AUDIO_MANIFEST_PATH,
    DEFAULT_SCENARIO_PATH,
)
from src.evaluation.benchmark.splits import DEFAULT_SPLIT_MANIFEST_PATH

DEFAULT_COVERAGE_TARGET_PATH = data_path("coverage_targets_v0.1.json")


def build_coverage_plan(
    scenario_path: str | Path = DEFAULT_SCENARIO_PATH,
    split_path: str | Path | None = DEFAULT_SPLIT_MANIFEST_PATH,
    audio_manifest_path: str | Path | None = DEFAULT_AUDIO_MANIFEST_PATH,
    target_path: str | Path = DEFAULT_COVERAGE_TARGET_PATH,
    *,
    profile: str = "public_beta",
) -> dict[str, Any]:
    """Compare current scenario coverage against a target profile."""
    suite = _load_json(scenario_path)
    targets = _load_json(target_path)
    profiles = targets.get("profiles", {})
    if profile not in profiles:
        known = ", ".join(sorted(profiles))
        raise ValueError(f"unknown coverage profile {profile!r}; expected one of: {known}")

    scenarios = suite.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise ValueError("scenario suite must contain a scenarios list")
    target = profiles[profile]
    split_manifest = _load_json(split_path) if split_path else None
    audio_manifest = _load_json(audio_manifest_path) if audio_manifest_path else None
    stats = _release_counts(scenarios, split_manifest, audio_manifest)
    gaps = {
        "total": _gap_value(stats["num_scenarios"], target.get("min_scenarios", 0)),
        "domains": _gap_map(stats["domains"], target.get("domains", {})),
        "tracks": _gap_map(stats["tracks"], target.get("tracks", {})),
        "difficulty": _gap_map(stats["difficulty"], target.get("difficulty", {})),
        "splits": _gap_map(stats["splits"], target.get("splits", {})),
        "audio_variants": _audio_gaps(stats["audio_variants"], target.get("audio_variants", {})),
    }
    return {
        "benchmark": "OpenVoiceCS-Bench",
        "profile": profile,
        "target_version": targets.get("version"),
        "passed": _gaps_clear(gaps),
        "current": stats,
        "targets": target,
        "gaps": gaps,
        "recommended_next_scenarios": _recommend_next_scenarios(gaps),
    }


def _release_counts(
    scenarios: list[dict[str, Any]],
    split_manifest: dict[str, Any] | None,
    audio_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    split_counts: Counter[str] = Counter()
    audio_split_counts: Counter[str] = Counter()
    if isinstance(split_manifest, dict):
        splits = split_manifest.get("splits", {})
        if isinstance(splits, dict):
            for split_name, split in splits.items():
                if isinstance(split, dict):
                    split_counts[str(split_name)] = len(split.get("scenario_ids", []))
                    audio_split_counts[str(split_name)] = len(split.get("audio_variant_ids", []))
    audio_variants = []
    if isinstance(audio_manifest, dict) and isinstance(audio_manifest.get("variants"), list):
        audio_variants = [
            variant for variant in audio_manifest["variants"]
            if isinstance(variant, dict)
        ]
    return {
        "num_scenarios": len(scenarios),
        "domains": dict(sorted(Counter(s.get("domain", "unknown") for s in scenarios).items())),
        "tracks": dict(sorted(Counter(s.get("track", "unknown") for s in scenarios).items())),
        "difficulty": dict(
            sorted(Counter(s.get("difficulty", "unknown") for s in scenarios).items())
        ),
        "splits": dict(sorted(split_counts.items())),
        "audio_variants": {
            "total": len(audio_variants),
            "tracks": dict(
                sorted(Counter(v.get("track", "unknown") for v in audio_variants).items())
            ),
            "splits": dict(sorted(audio_split_counts.items())),
        },
    }


def _gap_value(current: int, target: int) -> dict[str, int]:
    return {"current": current, "target": target, "needed": max(target - current, 0)}


def _gap_map(current: dict[str, int], target: dict[str, int]) -> dict[str, dict[str, int]]:
    gaps = {}
    for key, target_value in sorted(target.items()):
        current_value = current.get(key, 0)
        gaps[key] = _gap_value(current_value, target_value)
    return gaps


def _audio_gaps(current: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    return {
        "total": _gap_value(current.get("total", 0), target.get("total", 0)),
        "tracks": _gap_map(current.get("tracks", {}), target.get("tracks", {})),
        "splits": _gap_map(current.get("splits", {}), target.get("splits", {})),
    }


def _gaps_clear(gaps: dict[str, Any]) -> bool:
    if gaps["total"]["needed"]:
        return False
    for group_name in ("domains", "tracks", "difficulty", "splits"):
        if any(item["needed"] for item in gaps[group_name].values()):
            return False
    if gaps["audio_variants"]["total"]["needed"]:
        return False
    for group_name in ("tracks", "splits"):
        if any(item["needed"] for item in gaps["audio_variants"][group_name].values()):
            return False
    return True


def _recommend_next_scenarios(gaps: dict[str, Any]) -> list[dict[str, Any]]:
    recommendations = []
    remaining_total = max(
        gaps["total"]["needed"],
        max((item["needed"] for item in gaps["domains"].values()), default=0),
    )
    track_gaps = {key: dict(value) for key, value in gaps["tracks"].items()}
    difficulty_gaps = {key: dict(value) for key, value in gaps["difficulty"].items()}
    split_gaps = {key: dict(value) for key, value in gaps["splits"].items()}
    for domain, domain_gap in gaps["domains"].items():
        if remaining_total <= 0:
            break
        for _ in range(min(domain_gap["needed"], remaining_total)):
            recommendations.append(_recommendation_item(domain, track_gaps, difficulty_gaps, split_gaps))
            remaining_total -= 1
    while remaining_total > 0:
        domain = _stable_default_key(gaps["domains"])
        if domain is None:
            break
        recommendations.append(_recommendation_item(domain, track_gaps, difficulty_gaps, split_gaps))
        remaining_total -= 1
    return recommendations


def _recommendation_item(
    domain: str,
    track_gaps: dict[str, dict[str, int]],
    difficulty_gaps: dict[str, dict[str, int]],
    split_gaps: dict[str, dict[str, int]],
) -> dict[str, Any]:
    track = _largest_gap_key(track_gaps) or _stable_default_key(track_gaps)
    difficulty = _largest_gap_key(difficulty_gaps) or _stable_default_key(difficulty_gaps)
    split = _largest_gap_key(split_gaps) or _stable_default_key(split_gaps)
    item = {
        "domain": domain,
        "track": track,
        "difficulty": difficulty,
        "split": split,
        "count": 1,
    }
    _decrement_gap(track_gaps, track)
    _decrement_gap(difficulty_gaps, difficulty)
    _decrement_gap(split_gaps, split)
    return item


def _largest_gap_key(gaps: dict[str, dict[str, int]]) -> str | None:
    positive = [(key, item["needed"]) for key, item in gaps.items() if item["needed"] > 0]
    if not positive:
        return None
    return sorted(positive, key=lambda item: (-item[1], item[0]))[0][0]


def _stable_default_key(gaps: dict[str, dict[str, int]]) -> str | None:
    return sorted(gaps)[0] if gaps else None


def _decrement_gap(gaps: dict[str, dict[str, int]], key: str | None) -> None:
    if key is None:
        return
    gaps[key]["needed"] = max(gaps[key]["needed"] - 1, 0)


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return data
