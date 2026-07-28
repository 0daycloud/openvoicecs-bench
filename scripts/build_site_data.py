#!/usr/bin/env python3
"""Build ``site/data.json`` from release artifacts and the published leaderboard.

The static site renders every data-bearing number and table row from this file
at page load; nothing numeric is hand-typed into the HTML. The previous edition
of the site typed its numbers in as prose, which is how it came to claim wrong
scenario counts, a wrong split, and a leaderboard that silently dropped rows
from its own source CSV.

Counts are derived from the artifacts themselves — the scenario suite, the
split commitments, the release audit, the reference baselines, the frontier
report — and the leaderboard is read from the run's ``leaderboard.csv`` at run
time, never embedded, so a fresh sweep only requires re-running this script.

    python scripts/build_site_data.py
    python scripts/build_site_data.py --summary path/to/leaderboard.csv --output site/data.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "openvoicecs"

# The published sweep. Reports written by the current scorer need no re-scoring,
# so the leaderboard is built straight from the run directory.
DEFAULT_RUN = DATA / "runs" / "text_action_v02_merged"
DEFAULT_SUMMARY = DEFAULT_RUN / "leaderboard.csv"
DEFAULT_OUTPUT = ROOT / "site" / "data.json"

# Ranking policy, mirrored from scripts/build_leaderboard.py --min-coverage:
# a model is ranked only when >=90% of its trials produced a usable trace.
MIN_TRIAL_COVERAGE = 0.9
# Scorer weight of factual_grounding in overall_score (0-100 scale). Used to
# measure how sensitive the ordering is to the provisional phrase-matched
# grounding metric (see docs/leaderboard.md "How to read this").
GROUNDING_WEIGHT = 0.20

# Presentation order for tracks; anything unexpected sorts after, by name.
TRACK_ORDER = [
    "text_to_action",
    "audio_to_action",
    "end_to_end_voice",
    "robustness",
    "adversarial_compliance",
]

LEADERBOARD_METRICS = [
    "task_success",
    "tool_correctness",
    "sop_compliance",
    "privacy",
    "auth_integrity",
    "safety",
    "factual_grounding",
    "pass_at_k",
    "mean_pass_rate",
    "trial_coverage",
    "median_latency_ms",
    "avg_tool_calls",
    "tokens_per_success",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def count_by(scenarios: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for scenario in scenarios:
        value = scenario.get(key) or "unknown"
        counts[value] = counts.get(value, 0) + 1
    return counts


def build_corpus(scenario_suite: dict[str, Any], commitments: dict[str, Any]) -> dict[str, Any]:
    scenarios = scenario_suite["scenarios"]
    tracks = count_by(scenarios, "track")
    domains = count_by(scenarios, "domain")

    def track_key(track_id: str) -> tuple[int, str]:
        try:
            return (TRACK_ORDER.index(track_id), track_id)
        except ValueError:
            return (len(TRACK_ORDER), track_id)

    splits = {
        name: {
            "scenarios": split["num_scenarios"],
            "audio_variants": split["num_audio_variants"],
        }
        for name, split in commitments["splits"].items()
    }
    return {
        "num_scenarios": len(scenarios),
        "num_tracks": len(tracks),
        "num_domains": len(domains),
        "tracks": [
            {"id": track_id, "scenarios": tracks[track_id]}
            for track_id in sorted(tracks, key=track_key)
        ],
        "domains": [
            {"id": domain, "scenarios": count}
            for domain, count in sorted(domains.items(), key=lambda item: (-item[1], item[0]))
        ],
        "splits": splits,
    }


def build_release(audit: dict[str, Any], readiness: dict[str, Any]) -> dict[str, Any]:
    gates = {name: ok for name, ok in audit["release_gates"].items() if name != "passed"}
    validation = audit["validation"]
    return {
        "benchmark_version": audit["version"],
        "release_stage": audit["release_stage"],
        "generated_at": audit["generated_at"],
        "validation_passed": validation["passed"],
        "num_validation_issues": validation["num_issues"],
        "gates_total": len(gates),
        "gates_passed": sum(1 for ok in gates.values() if ok),
        "all_gates_passed": audit["release_gates"]["passed"],
        "readiness": {
            "profile": readiness["profile"],
            "passed": readiness["passed"],
            "num_issues": readiness["num_issues"],
        },
        "review_approval_coverage": audit["review_stats"]["scenario_approval_coverage"],
        "provenance_scenario_coverage": audit["provenance_stats"]["scenario_coverage"],
        "provenance_audio_coverage": audit["provenance_stats"]["audio_variant_coverage"],
        "audio_speaker_consent_rate": audit["provenance_stats"]["audio_speaker_consent_rate"],
        "open_errata": audit["changelog_stats"]["num_open_errata"],
        "num_audio_variants": audit["audio_manifest_stats"]["num_variants"],
        "audio_files_verified": audit["audio_asset_stats"]["num_sha256_verified"],
        "audio_perturbations": [
            {"id": name, "variants": count}
            for name, count in sorted(
                audit["audio_manifest_stats"]["perturbation_types"].items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
    }


def build_frontier(report: dict[str, Any]) -> dict[str, Any]:
    systems = []
    for system_id, card in sorted(report["scorecards"].items()):
        ci = card.get("axis_confidence_intervals", {}).get("task_success_rate", {})
        samples = card.get("latency_measurement", {}).get("sample_count")
        systems.append(
            {
                "id": system_id,
                "input": "audio" if "audio" in system_id else "text",
                "scenarios": card["num_scenarios"],
                "trials": samples,
                "task_success": card["task_success_rate"],
                "ci_low": ci.get("low"),
                "ci_high": ci.get("high"),
                "p50_ttfb_ms": card["p50_v2v_ttfb_ms"],
                "p95_ttfb_ms": card["p95_v2v_ttfb_ms"],
                "p95_last_byte_ms": card["p95_v2v_last_byte_ms"],
                "cost_per_success_usd": card["cost_usd_per_successful_conversation"],
                "experience": card["experience_score"],
            }
        )
    return {
        "generated_at": report["generated_at"],
        "concurrency_levels": report["environment"]["concurrency_levels"],
        "p95_ttfb_ms": max(system["p95_ttfb_ms"] for system in systems),
        "systems": systems,
    }


def build_anchors(baselines: dict[str, Any]) -> dict[str, Any]:
    anchors: dict[str, Any] = {}
    for baseline in baselines["baselines"]:
        if baseline["mode"] != "text":
            continue
        expected = baseline["expected"]
        anchors[baseline["agent"]] = {
            "id": baseline["id"],
            "description": baseline["description"],
            "overall_score": expected["overall_score"],
            "pass_k": expected["pass_k"],
            "task_success": expected["task_success"],
            "safety": expected["safety"],
            "num_scenarios": expected["num_scenarios"],
            "trials_per_scenario": expected["num_trials_per_scenario"],
        }
    for agent in ("oracle", "noop"):
        if agent not in anchors:
            raise SystemExit(f"reference baselines are missing the text-mode `{agent}` anchor")
    return anchors


def _number(value: str | None, fallback: float = 0.0) -> float:
    """Parse a CSV cell, treating a blank as absent telemetry rather than zero-ish.

    Providers that report no token usage or no pricing leave these columns empty;
    crashing on that would make the site un-buildable for a perfectly valid run.
    """
    if value is None or value == "":
        return fallback
    try:
        return float(value)
    except ValueError:
        return fallback


def read_summary(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise SystemExit(f"{path} contains no leaderboard rows")
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        row: dict[str, Any] = {"model_id": raw["model_id"]}
        row["overall_score"] = _number(raw.get("overall_score"))
        for metric in LEADERBOARD_METRICS:
            row[metric] = _number(raw.get(metric))
        for metric in ("scored_trials", "infrastructure_error_trials", "measured_scenarios"):
            row[metric] = int(_number(raw.get(metric)))
        rows.append(row)
    return rows


def without_grounding(row: dict[str, Any]) -> float:
    return row["overall_score"] - GROUNDING_WEIGHT * 100.0 * row["factual_grounding"]


def grounding_sensitivity(ranked: list[dict[str, Any]]) -> tuple[bool, int, bool]:
    """How much of the ordering is an artifact of the provisional grounding metric.

    ``factual_grounding`` is a literal phrase matcher (see
    docs/known-limitations.md). Re-sorting with its weighted contribution
    removed tells us three things: whether the top two swap (present them as
    tied if so), how many ranks move at all, and whether the podium survives.
    """
    if len(ranked) < 2:
        return False, 0, True
    alt = sorted(ranked, key=lambda row: -without_grounding(row))
    top_tie = without_grounding(ranked[0]) < without_grounding(ranked[1])
    moved = sum(1 for pos, row in enumerate(ranked) if alt[pos] is not row)
    podium = [row["model_id"] for row in ranked[:3]] == [row["model_id"] for row in alt[:3]]
    return top_tie, moved, podium


def build_leaderboard(summary_path: Path, corpus: dict[str, Any]) -> dict[str, Any]:
    all_rows = read_summary(summary_path)
    ranked = [row for row in all_rows if row["trial_coverage"] >= MIN_TRIAL_COVERAGE]
    ranked.sort(key=lambda row: -row["overall_score"])
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["tied"] = False
    tied, moved, podium_stable = grounding_sensitivity(ranked)
    if tied:
        ranked[0]["tied"] = ranked[1]["tied"] = True

    track = "text_to_action"
    track_scenarios = next(
        (entry["scenarios"] for entry in corpus["tracks"] if entry["id"] == track), None
    )
    total_trials = max(row["scored_trials"] + row["infrastructure_error_trials"] for row in ranked)

    # `raw_summary.csv` beside the leaderboard lists every attempted model,
    # including the ones that never produced enough usable trials to rank.
    run_dir = summary_path.resolve().parent
    raw_summary = run_dir / "raw_summary.csv"
    if raw_summary.is_file():
        with raw_summary.open(encoding="utf-8", newline="") as handle:
            models_attempted = sum(1 for _ in csv.DictReader(handle)) or None
    else:
        reports_dir = run_dir / "reports"
        models_attempted = (
            len(list(reports_dir.glob("*.json"))) if reports_dir.is_dir() else 0
        ) or None

    try:
        source = str(summary_path.resolve().relative_to(ROOT))
    except ValueError:
        source = str(summary_path)

    return {
        "source": source,
        "source_run": run_dir.name,
        "scorer_version": "v0.2",
        "track": track,
        "track_scenarios": track_scenarios,
        "trials_per_scenario": round(total_trials / track_scenarios) if track_scenarios else None,
        "min_trial_coverage": MIN_TRIAL_COVERAGE,
        "grounding_weight": GROUNDING_WEIGHT,
        "models_attempted": models_attempted,
        "ranked_models": len(ranked),
        "excluded_models": (models_attempted - len(ranked)) if models_attempted else None,
        "grounding_moved_models": moved,
        "podium_stable": podium_stable,
        "overall_floor": ranked[-1]["overall_score"],
        "grounding_min": min(row["factual_grounding"] for row in ranked),
        "grounding_max": max(row["factual_grounding"] for row in ranked),
        "top_tie": tied,
        "rows": ranked,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="leaderboard CSV to publish (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="where to write the site data file (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    if not args.summary.is_file():
        raise SystemExit(f"summary CSV not found: {args.summary}")

    scenario_suite = load_json(DATA / "scenarios_v0.1.json")
    commitments = load_json(DATA / "split_commitments_v0.1.json")
    audit = load_json(DATA / "release_audit.json")
    readiness = load_json(DATA / "readiness_leaderboard_v1.json")
    baselines = load_json(DATA / "baselines" / "reference_baselines_v0.1.json")
    frontier = load_json(DATA / "releases" / "frontier_seed" / "frontier_report.json")

    corpus = build_corpus(scenario_suite, commitments)
    release = build_release(audit, readiness)
    corpus["num_audio_variants"] = release.pop("num_audio_variants")
    corpus["audio_perturbations"] = release.pop("audio_perturbations")

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "corpus": corpus,
        "release": release,
        "frontier": build_frontier(frontier),
        "anchors": (anchors := build_anchors(baselines)),
        "leaderboard": (board := build_leaderboard(args.summary, corpus)),
    }
    noop_floor = anchors["noop"]["overall_score"]
    board["models_below_noop"] = sum(
        1 for row in board["rows"] if row["overall_score"] < noop_floor
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    board = payload["leaderboard"]
    print(
        f"wrote {args.output}: {corpus['num_scenarios']} scenarios, "
        f"{board['ranked_models']} ranked models "
        f"(of {board['models_attempted']} attempted), "
        f"top tie: {board['top_tie']}, "
        f"grounding moves {board['grounding_moved_models']} ranks"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
