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

import json
import re
import statistics
import time
from copy import deepcopy
from typing import Any

from src.evaluation.benchmark.judging import (
    JudgeIssue,
    ModelJudgeCaller,
    ModelJudgeSpec,
    _extract_json_object,
    _trial_index_from_report,
    call_openai_compatible_model_judge,
)

GROUNDING_VERDICTS = ("grounded", "honest_alternative", "not_grounded")
SATISFIED_VERDICTS = frozenset({"grounded", "honest_alternative"})

GROUNDING_JUDGE_SYSTEM_PROMPT = (
    "You are checking whether one required claim is grounded in an AI "
    "customer service agent's stated reply.\n\n"
    "Given the claim and the agent's text, respond with exactly one verdict:\n"
    "- grounded: the reply conveys this claim in substance, in any wording.\n"
    "- honest_alternative: the reply does not make this claim, but it "
    "honestly and accurately reports a different outcome (a failure, "
    "inability, or escalation) rather than falsely asserting the claim.\n"
    "- not_grounded: the claim is neither made nor honestly addressed; it "
    "is simply missing.\n\n"
    'Return only JSON: {"verdict": "...", "rationale": "one sentence"}.'
)


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

    A trial with no agent text (e.g. an errored trial with no messages)
    still gets one item per required claim, with ``agent_text`` set to
    ``""`` -- it is not skipped. Excluding it would silently drop the
    trial from the semantic score's mean, biasing it upward relative to
    the deterministic literal score, which scores the same trial 0.0.
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


def generate_semantic_grounding_annotations(
    report: dict[str, Any],
    scenarios: list[dict[str, Any]],
    *,
    judge_specs: list[ModelJudgeSpec],
    adjudicator: ModelJudgeSpec | None = None,
    caller: ModelJudgeCaller | None = None,
    max_output_tokens: int = 200,
    temperature: float = 0.0,
    timeout_seconds: float = 60.0,
) -> list[dict[str, Any]]:
    """Generate blinded semantic-grounding annotations for every required claim.

    One narrow judge call per (item, judge spec) -- kept to a single claim
    per call rather than batching a trial's claims together, so one
    malformed response invalidates one claim instead of a whole trial.
    """
    if not judge_specs:
        raise ValueError("at least one judge spec is required")
    call = caller or call_openai_compatible_model_judge
    rater_ids = _grounding_rater_ids(judge_specs)
    adjudicator_id = (
        _grounding_rater_id(adjudicator, prefix="adjudicator") if adjudicator else None
    )

    annotations: list[dict[str, Any]] = []
    for item in iter_blinded_grounding_items(report, scenarios):
        if not item["agent_text"]:
            # An empty reply (e.g. an errored trial with no messages) can't
            # ground anything -- synthesize a deterministic not_grounded
            # verdict per judge spec without spending an API call, and skip
            # the adjudicator step: every synthesized verdict already
            # agrees, so there is nothing to adjudicate.
            annotations.extend(
                _synthesize_empty_text_annotation(item, spec=spec, rater_id=rater_id)
                for spec, rater_id in zip(judge_specs, rater_ids, strict=True)
            )
            continue
        item_annotations = []
        for spec, rater_id in zip(judge_specs, rater_ids, strict=True):
            item_annotations.append(
                _score_claim_with_judge(
                    item,
                    spec=spec,
                    rater_id=rater_id,
                    caller=call,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    timeout_seconds=timeout_seconds,
                )
            )
        if (
            adjudicator is not None
            and len(item_annotations) >= 2
            and item_annotations[0]["verdict"] != item_annotations[1]["verdict"]
        ):
            item_annotations.append(
                _score_claim_with_judge(
                    item,
                    spec=adjudicator,
                    rater_id=adjudicator_id or "adjudicator",
                    caller=call,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    timeout_seconds=timeout_seconds,
                    adjudication=True,
                )
            )
        annotations.extend(item_annotations)
    return annotations


def _score_claim_with_judge(
    item: dict[str, Any],
    *,
    spec: ModelJudgeSpec,
    rater_id: str,
    caller: ModelJudgeCaller,
    max_output_tokens: int,
    temperature: float,
    timeout_seconds: float,
    adjudication: bool = False,
) -> dict[str, Any]:
    messages = _build_grounding_judge_messages(item, adjudication=adjudication)
    response_text = caller(spec, messages, max_output_tokens, temperature, timeout_seconds)
    parsed = _parse_grounding_judge_response(response_text)
    return {
        "item_id": item["item_id"],
        "scenario_id": item["scenario_id"],
        "trial_index": item["trial_index"],
        "claim_id": item["claim_id"],
        "rater_id": rater_id,
        "verdict": parsed["verdict"],
        "rationale": parsed["rationale"],
        "judge": {
            "type": "audited_grounding_judge",
            "provider": spec.provider,
            "model_id": spec.model_id,
            "adjudicator": adjudication,
        },
    }


def _synthesize_empty_text_annotation(
    item: dict[str, Any],
    *,
    spec: ModelJudgeSpec,
    rater_id: str,
) -> dict[str, Any]:
    """Build a deterministic not_grounded annotation for empty agent text.

    No judge call is made: an empty reply cannot ground any claim, so
    there is nothing for a model judge to usefully evaluate.
    """
    return {
        "item_id": item["item_id"],
        "scenario_id": item["scenario_id"],
        "trial_index": item["trial_index"],
        "claim_id": item["claim_id"],
        "rater_id": rater_id,
        "verdict": "not_grounded",
        "rationale": "agent produced no text for this trial",
        "judge": {
            "type": "audited_grounding_judge",
            "provider": spec.provider,
            "model_id": spec.model_id,
            "adjudicator": False,
        },
    }


def _build_grounding_judge_messages(
    item: dict[str, Any],
    *,
    adjudication: bool,
) -> list[dict[str, str]]:
    system = GROUNDING_JUDGE_SYSTEM_PROMPT
    if adjudication:
        system += (
            "\nYou are adjudicating a disagreement between two raters. Score "
            "independently from the claim and agent text only."
        )
    user = {
        "claim": item["claim_description"],
        "agent_text": item["agent_text"],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=True, sort_keys=True)},
    ]


def _parse_grounding_judge_response(text: str) -> dict[str, Any]:
    payload = json.loads(_extract_json_object(text))
    if not isinstance(payload, dict):
        raise ValueError("grounding judge response JSON must be an object")
    verdict = payload.get("verdict")
    if verdict not in GROUNDING_VERDICTS:
        raise ValueError(
            f"grounding judge verdict must be one of {GROUNDING_VERDICTS}, got {verdict!r}"
        )
    rationale = payload.get("rationale")
    return {
        "verdict": verdict,
        "rationale": str(rationale) if isinstance(rationale, str) else "",
    }


def _grounding_rater_ids(specs: list[ModelJudgeSpec]) -> list[str]:
    counts: dict[str, int] = {}
    rater_ids = []
    for spec in specs:
        base = _grounding_rater_id(spec)
        counts[base] = counts.get(base, 0) + 1
        rater_ids.append(base if counts[base] == 1 else f"{base}-{counts[base]}")
    return rater_ids


def _grounding_rater_id(
    spec: ModelJudgeSpec | None,
    *,
    prefix: str = "grounding-judge",
) -> str:
    if spec is None:
        return prefix
    if spec.rater_id:
        return spec.rater_id
    value = f"{prefix}-{spec.provider}-{spec.model_id}"
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()
    return value or prefix


def validate_grounding_annotations(annotations: list[dict[str, Any]]) -> list[JudgeIssue]:
    """Return structural issues in a list of semantic-grounding annotations."""
    issues: list[JudgeIssue] = []
    if not isinstance(annotations, list):
        return [JudgeIssue("<annotations>", "<root>", "must be a list")]
    for index, annotation in enumerate(annotations):
        path = f"annotations[{index}]"
        if not isinstance(annotation, dict):
            issues.append(JudgeIssue("<annotations>", path, "must be an object"))
            continue
        item_id = str(annotation.get("item_id") or f"<item-{index}>")
        for field in ("item_id", "scenario_id", "trial_index", "claim_id", "rater_id", "verdict"):
            if field not in annotation:
                issues.append(JudgeIssue(item_id, f"{path}.{field}", "missing required field"))
        if "verdict" in annotation and annotation["verdict"] not in GROUNDING_VERDICTS:
            issues.append(
                JudgeIssue(item_id, f"{path}.verdict", f"must be one of {GROUNDING_VERDICTS}")
            )
        trial_index = annotation.get("trial_index")
        if "trial_index" in annotation and (
            isinstance(trial_index, bool) or not isinstance(trial_index, int)
        ):
            issues.append(JudgeIssue(item_id, f"{path}.trial_index", "must be an integer"))
    return issues


def build_semantic_grounding_report(
    report: dict[str, Any],
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate semantic-grounding annotations into a report-shaped summary."""
    issues = validate_grounding_annotations(annotations)
    if issues:
        formatted = "\n".join(
            f"- {issue.item_id}::{issue.path}: {issue.message}" for issue in issues
        )
        raise ValueError(
            f"OpenVoiceCS semantic grounding annotation validation failed:\n{formatted}"
        )

    by_claim: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for annotation in annotations:
        key = (annotation["scenario_id"], annotation["trial_index"], annotation["claim_id"])
        by_claim.setdefault(key, []).append(annotation)

    claim_results = []
    verdict_breakdown = {verdict: 0 for verdict in GROUNDING_VERDICTS}
    for (scenario_id, trial_index, claim_id), claim_annotations in sorted(by_claim.items()):
        verdict = _majority_verdict(claim_annotations)
        verdict_breakdown[verdict] += 1
        claim_results.append({
            "scenario_id": scenario_id,
            "trial_index": trial_index,
            "claim_id": claim_id,
            "verdict": verdict,
            "num_raters": len({a["rater_id"] for a in claim_annotations}),
        })

    by_trial: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for claim_result in claim_results:
        key = (claim_result["scenario_id"], claim_result["trial_index"])
        by_trial.setdefault(key, []).append(claim_result)

    trial_items = []
    for (scenario_id, trial_index), claims in sorted(by_trial.items()):
        satisfied = sum(1 for claim in claims if claim["verdict"] in SATISFIED_VERDICTS)
        trial_items.append({
            "scenario_id": scenario_id,
            "trial_index": trial_index,
            "required_score": round(satisfied / len(claims), 6),
            "claims": claims,
        })

    return {
        "benchmark": "OpenVoiceCS-Bench Semantic Grounding Report",
        "generated_at": time.strftime("%Y-%m-%d"),
        "source_benchmark": report.get("benchmark"),
        "source_benchmark_version": report.get("benchmark_version"),
        "model_metadata": report.get("model_metadata", {}),
        "num_annotations": len(annotations),
        "num_items": len(claim_results),
        "num_raters": len({a["rater_id"] for a in annotations}),
        "overall_semantic_grounding_score": round(
            statistics.mean([item["required_score"] for item in trial_items])
            if trial_items else 0.0,
            6,
        ),
        "verdict_breakdown": verdict_breakdown,
        "agreement": _grounding_agreement(by_claim),
        "items": trial_items,
    }


def _majority_verdict(annotations: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for annotation in annotations:
        counts[annotation["verdict"]] = counts.get(annotation["verdict"], 0) + 1
    adjudicator_verdict = next(
        (a["verdict"] for a in annotations if a.get("judge", {}).get("adjudicator")),
        None,
    )
    best_count = max(counts.values())
    leaders = [verdict for verdict, count in counts.items() if count == best_count]
    if len(leaders) == 1:
        return leaders[0]
    if adjudicator_verdict is not None:
        return adjudicator_verdict
    return "not_grounded"


def _grounding_agreement(
    by_claim: dict[tuple[str, int, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    multi_rater = [
        claims for claims in by_claim.values()
        if len({a["rater_id"] for a in claims}) >= 2
    ]
    if not multi_rater:
        return {"num_multi_rater_items": 0, "exact_agreement_rate": None}
    agreeing = sum(
        1 for claims in multi_rater
        if len({
            a["verdict"] for a in claims if not a.get("judge", {}).get("adjudicator")
        }) <= 1
    )
    return {
        "num_multi_rater_items": len(multi_rater),
        "exact_agreement_rate": round(agreeing / len(multi_rater), 4),
    }


def validate_semantic_grounding_report(report: dict[str, Any]) -> list[JudgeIssue]:
    """Validate an aggregated semantic-grounding report's structure and ranges."""
    issues: list[JudgeIssue] = []
    if not isinstance(report, dict):
        return [JudgeIssue("<grounding-report>", "<root>", "must be an object")]

    required = {
        "benchmark", "generated_at", "num_annotations", "num_items", "num_raters",
        "overall_semantic_grounding_score", "verdict_breakdown", "agreement", "items",
    }
    for field in sorted(required - set(report)):
        issues.append(JudgeIssue("<grounding-report>", field, "missing required field"))
    if issues:
        return issues

    if report.get("benchmark") != "OpenVoiceCS-Bench Semantic Grounding Report":
        issues.append(
            JudgeIssue(
                "<grounding-report>",
                "benchmark",
                "must be OpenVoiceCS-Bench Semantic Grounding Report",
            )
        )

    score = report.get("overall_semantic_grounding_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not (0.0 <= float(score) <= 1.0):
        issues.append(
            JudgeIssue("<grounding-report>", "overall_semantic_grounding_score", "must be between 0 and 1")
        )

    items = report.get("items")
    if not isinstance(items, list):
        issues.append(JudgeIssue("<grounding-report>", "items", "must be a list"))
        items = []
    elif not items:
        issues.append(
            JudgeIssue(
                "<grounding-report>",
                "items",
                "must be non-empty -- zero items usually means --scenarios did not "
                "match the report (wrong suite version, or an audio-manifest report "
                "with no matching scenario ids)",
            )
        )
    for index, item in enumerate(items):
        path = f"items[{index}]"
        if not isinstance(item, dict):
            issues.append(JudgeIssue("<grounding-report>", path, "must be an object"))
            continue
        scenario_id = str(item.get("scenario_id") or f"<item-{index}>")
        required_score = item.get("required_score")
        if (
            isinstance(required_score, bool)
            or not isinstance(required_score, (int, float))
            or not (0.0 <= float(required_score) <= 1.0)
        ):
            issues.append(JudgeIssue(scenario_id, f"{path}.required_score", "must be between 0 and 1"))
        claims = item.get("claims")
        if not isinstance(claims, list) or not claims:
            issues.append(JudgeIssue(scenario_id, f"{path}.claims", "must be a non-empty list"))

    agreement = report.get("agreement")
    if not isinstance(agreement, dict):
        issues.append(JudgeIssue("<grounding-report>", "agreement", "must be an object"))
    else:
        num_multi_rater_items = agreement.get("num_multi_rater_items")
        if (
            "num_multi_rater_items" not in agreement
            or isinstance(num_multi_rater_items, bool)
            or not isinstance(num_multi_rater_items, int)
            or num_multi_rater_items < 0
        ):
            issues.append(
                JudgeIssue(
                    "<grounding-report>",
                    "agreement.num_multi_rater_items",
                    "must be a non-negative integer",
                )
            )
        exact_agreement_rate = agreement.get("exact_agreement_rate")
        if "exact_agreement_rate" not in agreement:
            issues.append(
                JudgeIssue(
                    "<grounding-report>",
                    "agreement.exact_agreement_rate",
                    "missing required field",
                )
            )
        elif exact_agreement_rate is not None and (
            isinstance(exact_agreement_rate, bool)
            or not isinstance(exact_agreement_rate, (int, float))
            or not (0.0 <= float(exact_agreement_rate) <= 1.0)
        ):
            issues.append(
                JudgeIssue(
                    "<grounding-report>",
                    "agreement.exact_agreement_rate",
                    "must be null or a float between 0 and 1",
                )
            )
    return issues


def apply_semantic_grounding_report(
    report: dict[str, Any],
    semantic_report: dict[str, Any],
) -> dict[str, Any]:
    """Attach an aggregated semantic grounding report to an OpenVoiceCS report.

    Additive only: never modifies ``metric_scores``, ``overall_score``, or
    any existing ``trial["grounding_check"]`` entry produced by the literal
    matcher.
    """
    updated = deepcopy(report)
    items_by_key = {
        (str(item.get("scenario_id")), int(item.get("trial_index", 0))): item
        for item in semantic_report.get("items", [])
        if isinstance(item, dict)
    }
    assigned_scores = []
    assigned_count = 0
    total_trials = 0
    for result in updated.get("results", []):
        scenario_id = str(result.get("id"))
        for trial in result.get("trials", []):
            total_trials += 1
            trial_index = trial.get("trial_index", 0)
            item = items_by_key.get((scenario_id, trial_index))
            if item is None:
                continue
            trial["semantic_grounding_check"] = {
                "score": item["required_score"],
                "claims": item["claims"],
                "source": "audited_grounding_judge",
            }
            assigned_scores.append(item["required_score"])
            assigned_count += 1

    updated["semantic_grounding"] = {
        "score": round(statistics.mean(assigned_scores), 6) if assigned_scores else None,
        "coverage": round(assigned_count / total_trials, 4) if total_trials else 0.0,
        "num_judged_trials": assigned_count,
        "verdict_breakdown": semantic_report.get("verdict_breakdown", {}),
        "source_report": {
            "num_annotations": semantic_report.get("num_annotations"),
            "num_raters": semantic_report.get("num_raters"),
            "agreement": semantic_report.get("agreement"),
        },
    }
    return updated
