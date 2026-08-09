"""Metamorphic transforms for grader validation — measurement tooling, not scorer.

Known-answer probes (``grader_probe`` on PR #9, ``grader_eval`` on PR #10)
hand-write individual mutations of the oracle trace. This module instead
declares a small set of *relations* and generates cases mechanically across
the whole corpus from recorded model traces: a meaning-preserving transform
must not change the grounding score, a defect-injecting transform must drop
it. A violation is a grader bug provable without any ground-truth label —
which also sidesteps criterion contamination: a state-consuming grader cannot
be validated against state, but it can be validated against relations.

Every transform reports ``(scenario, trace)`` on success or ``(None, reason)``
when it cannot be grounded in the case — a transform that cannot prove it
injected (or avoided injecting) a defect must be counted as skipped, never as
a pass. Lives under ``tests/`` because it measures the grader rather than
being part of it; the scoped money detector it exercises is production code
in ``openvoicecs.py``.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from src.evaluation.benchmark.openvoicecs import monetary_values

#: transform name -> how the grounding score must respond
EXPECTED_RELATION = {
    "synonym_swap": "unchanged",     # vocabulary blindness
    "collide_amount": "drop",        # fabricated figure hidden by an identifier
    "corrupt_state": "drop",         # claim credited while the outcome is false
    "neutral_append": "unchanged",   # over-reaction to irrelevant true text
}

_SYNONYMS = (("completed", "wrapped up"), ("logged", "noted on your account"))
_NEUTRAL_SENTENCE = "I appreciate your patience while we sorted this out together today."
_ID_RE = re.compile(r"[a-z]+_(\d{3,})")
_MONEY_KEY_RE = re.compile(
    r"(amount|cents|credit|fee|price|total|balance|charge|refund)", re.IGNORECASE
)


def _agent_texts(trace: dict[str, Any]) -> list[str]:
    return [m.get("text", "") for m in trace.get("messages", []) if m.get("role") == "agent"]


def _joined_text(trace: dict[str, Any]) -> str:
    return " ".join(_agent_texts(trace))


def _matches_any(text: str, terms: list[str]) -> bool:
    return any(t and re.search(t, text, flags=re.IGNORECASE) for t in terms)


def _required_claims(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    return scenario.get("oracle", {}).get("grounding", {}).get("required_claims", [])


def _append_to_last_agent_turn(trace: dict[str, Any], sentence: str) -> dict[str, Any]:
    out = deepcopy(trace)
    for message in reversed(out.get("messages", [])):
        if message.get("role") == "agent":
            message["text"] = (message.get("text", "") + " " + sentence).strip()
            return out
    return out


def transform_synonym_swap(scenario, trace, state_passed, final_state):
    """Swap completion verbs for synonyms; meaning and truth are unchanged."""
    if not state_passed:
        return None, "state check failed: original claim is not truthfully credited"
    text = _joined_text(trace)
    claims = [c for c in _required_claims(scenario)
              if _matches_any(text, c.get("any_terms", []))
              and any(term in ("completed", "logged") for term in c.get("any_terms", []))]
    if not claims:
        return None, "no matched completion-vocabulary claim to paraphrase"
    out = deepcopy(trace)
    changed = False
    for message in out.get("messages", []):
        if message.get("role") != "agent":
            continue
        new = message.get("text", "")
        for word, synonym in _SYNONYMS:
            new = re.sub(rf"\b{word}\b", synonym, new, flags=re.IGNORECASE)
        changed |= new != message.get("text", "")
        message["text"] = new
    if not changed:
        return None, "completion verbs not present in agent text"
    return scenario, out


def transform_collide_amount(scenario, trace, state_passed, final_state):
    """Append a fabricated dollar figure whose digits collide with an id."""
    sandbox = json.dumps(
        [scenario.get("conversation"), scenario.get("initial_state"), final_state],
        sort_keys=True, default=str,
    )
    money = monetary_values(scenario, final_state)
    colliding = sorted(d for d in _ID_RE.findall(sandbox) if d not in money)
    if not colliding:
        return None, "no identifier digits distinct from every monetary value"
    sentence = f"The final figure comes to ${colliding[0]}."
    for claim in _required_claims(scenario):
        if _matches_any(sentence, claim.get("any_terms", [])):
            return None, "fabricated sentence would satisfy a required claim"
    return scenario, _append_to_last_agent_turn(trace, sentence)


def transform_corrupt_state(scenario, trace, state_passed, final_state):
    """Falsify one expected-state path; the claim text now asserts a non-event."""
    if not state_passed:
        return None, "state check already failed: no truthful credit to falsify"
    text = _joined_text(trace)
    if not any(_matches_any(text, c.get("any_terms", [])) for c in _required_claims(scenario)):
        return None, "no matched required claim whose credit could become false"
    expected = scenario.get("oracle", {}).get("expected_state") or {}
    node = deepcopy(expected)
    cursor = node
    while isinstance(cursor, dict) and cursor:
        key = sorted(cursor)[0]
        if not isinstance(cursor[key], dict) or not cursor[key]:
            cursor[key] = "__falsified_by_probe__"
            break
        cursor = cursor[key]
    else:
        return None, "scenario has no expected_state path to falsify"
    out = deepcopy(scenario)
    out["oracle"]["expected_state"] = node
    return out, trace


def transform_neutral_append(scenario, trace, state_passed, final_state):
    """Append a true, on-topic pleasantry that grounds no claim either way."""
    for claim in _required_claims(scenario):
        if _matches_any(_NEUTRAL_SENTENCE, claim.get("any_terms", [])):
            return None, "neutral sentence collides with a required-claim term"
    for claim in scenario.get("oracle", {}).get("grounding", {}).get("forbidden_claims", []):
        if _matches_any(_NEUTRAL_SENTENCE, claim.get("patterns", [])):
            return None, "neutral sentence collides with a forbidden-claim pattern"
    return scenario, _append_to_last_agent_turn(trace, _NEUTRAL_SENTENCE)


TRANSFORMS = {
    "synonym_swap": transform_synonym_swap,
    "collide_amount": transform_collide_amount,
    "corrupt_state": transform_corrupt_state,
    "neutral_append": transform_neutral_append,
}
