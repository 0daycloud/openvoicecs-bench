"""Offline, blinded semantic grounding check for OpenVoiceCS required claims.

Deterministic phrase-matched grounding (``check_factual_grounding`` in
openvoicecs.py) remains the default and only score that feeds
``metric_scores.factual_grounding`` / ``overall_score``. This module adds a
separate, auditable path: narrow LLM-judge calls that classify each required
claim as grounded, honestly addressed by a different truthful outcome, or
genuinely missing. It follows the same blinded/audited pattern as the
model-judge pipeline in judging.py, run offline and post-hoc against an
already-scored report rather than inline during scoring.
"""

from __future__ import annotations

from typing import Any

from src.evaluation.benchmark.judging import _trial_index_from_report

GROUNDING_VERDICTS = ("grounded", "honest_alternative", "not_grounded")
SATISFIED_VERDICTS = frozenset({"grounded", "honest_alternative"})


def iter_blinded_grounding_items(
    report: dict[str, Any],
    scenarios: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return one blinded item per (trial, required claim).

    ``report["results"]`` does not carry ``oracle.grounding.required_claims``
    -- only the literal matcher's ``missing_required_claims`` is present in
    each trial's ``grounding_check``, which is not the full claim set an
    independent judge needs. ``scenarios`` is the same suite the report was
    produced from (e.g. ``OpenVoiceCSBench.load(path).scenarios``), matched
    to report results by ``id`` -- stable for audio variants too, since
    ``build_audio_variant_scenarios`` copies the base scenario's oracle
    under the variant's own id.

    Items carry no scenario identity beyond the id needed to key results
    back together, no model identity, no oracle pass/fail, no tool calls,
    and no scores -- only the claim text and the agent's own words.
    """
    scenarios_by_id = {
        str(scenario.get("id")): scenario
        for scenario in scenarios
        if isinstance(scenario, dict) and scenario.get("id")
    }
    items: list[dict[str, Any]] = []
    for result in report.get("results", []):
        if not isinstance(result, dict):
            continue
        scenario_id = str(result.get("id") or "")
        scenario = scenarios_by_id.get(scenario_id)
        if scenario is None:
            continue
        required_claims = _required_claims(scenario)
        if not required_claims:
            continue
        trials = result.get("trials") if isinstance(result.get("trials"), list) else []
        for fallback_index, trial in enumerate(trials):
            if not isinstance(trial, dict):
                continue
            trial_index = _trial_index_from_report(trial, fallback_index)
            agent_text = _blinded_agent_text(trial.get("messages"))
            if not agent_text:
                continue
            for claim in required_claims:
                claim_id = str(claim.get("id") or "")
                if not claim_id:
                    continue
                items.append({
                    "item_id": f"{scenario_id}:{trial_index}:{claim_id}",
                    "scenario_id": scenario_id,
                    "trial_index": trial_index,
                    "claim_id": claim_id,
                    "claim_description": _claim_description(claim),
                    "agent_text": agent_text,
                })
    return items


def _required_claims(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    oracle = scenario.get("oracle")
    if not isinstance(oracle, dict):
        return []
    grounding = oracle.get("grounding")
    if not isinstance(grounding, dict):
        return []
    claims = grounding.get("required_claims")
    return claims if isinstance(claims, list) else []


def _claim_description(claim: dict[str, Any]) -> str:
    terms = claim.get("any_terms")
    if isinstance(terms, list) and terms:
        return " / ".join(str(term) for term in terms)
    return str(claim.get("id", ""))


def _blinded_agent_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    parts = [
        str(message.get("text", ""))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "agent"
    ]
    return " ".join(part for part in parts if part).strip()
