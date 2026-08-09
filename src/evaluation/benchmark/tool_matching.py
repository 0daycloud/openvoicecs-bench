"""Tool-argument matching and sandbox state writes, shared by every executor.

Both the scorer's ``replay_tool_calls`` and the stepwise action loop's
``_execute_scenario_tool`` decide whether a model's tool call is acceptable.
They must decide identically: the loop's verdict is the feedback the model gets
inside a bounded round budget, so a stricter loop burns rounds rejecting calls
the scorer would have accepted, and the lost trial is charged to the model.
These helpers are the single copy of that decision.
"""

from __future__ import annotations

import re
from typing import Any


def _normalize_argument_token(value: str) -> str:
    """Collapse a label to a comparable token.

    Scenario labels are authored in ``snake_case`` while models routinely emit
    the same label as prose (``"damaged furniture"`` for ``damaged_furniture``).
    Punctuation, case, and separator differences are presentation, not
    behaviour, so they are normalized away before comparison.
    """
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _values_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        return _normalize_argument_token(expected) == _normalize_argument_token(actual)
    return actual == expected


def _dict_contains(actual: dict[str, Any], expected_subset: dict[str, Any]) -> bool:
    for key, expected in expected_subset.items():
        if key not in actual:
            return False
        actual_value = actual[key]
        if isinstance(expected, dict) and isinstance(actual_value, dict):
            if not _dict_contains(actual_value, expected):
                return False
        elif not _values_match(expected, actual_value):
            return False
    return True


def _model_required_arguments(tool_def: dict[str, Any]) -> dict[str, Any]:
    """Required arguments the model itself is expected to supply.

    ``generated_arguments`` are system-assigned and ``argument_bindings`` are
    carried over from a prior call's result, so neither is something the model
    can be asked to know.
    """
    generated = set((tool_def.get("generated_arguments") or {}).keys())
    bound = set((tool_def.get("argument_bindings") or {}).keys())
    return {
        key: value
        for key, value in (tool_def.get("required_arguments") or {}).items()
        if key not in generated and key not in bound
    }


def _argument_binding_errors(
    tool_def: dict[str, Any],
    arguments: dict[str, Any],
    bindings: dict[str, Any],
) -> list[dict[str, Any]]:
    errors = []
    for argument, binding in (tool_def.get("argument_bindings") or {}).items():
        if argument not in bindings:
            errors.append({
                "argument": argument,
                "error": "binding_source_missing",
                "binding": binding,
            })
            continue
        actual = (arguments or {}).get(argument)
        # Compared like any other argument: echoing a bound id back in a
        # different case is a transcription difference, not a wrong value.
        if not _values_match(bindings[argument], actual):
            errors.append({
                "argument": argument,
                "error": "bound_value_not_used",
                "expected": bindings[argument],
                "actual": actual,
                "binding": binding,
            })
    return errors


def _effective_tool_arguments(
    tool_def: dict[str, Any],
    arguments: dict[str, Any],
    bindings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The arguments a call is judged on, after system-supplied values fill in.

    ``generated_arguments`` stand in for values the model cannot know, so they
    fill gaps — they do not overwrite. Overwriting would erase the difference
    between omitting a system-assigned field and deliberately supplying a
    different value for it, and that difference is exactly what a
    ``forbidden_tool_calls`` pattern is written to detect. A model that omits
    the field is still not penalized: ``_model_required_arguments`` keeps
    generated keys out of the required-argument check.
    """
    effective = dict(arguments or {})
    effective.update(bindings or {})
    for key, value in (tool_def.get("generated_arguments") or {}).items():
        effective.setdefault(key, value)
    return effective


def _set_path(data: dict[str, Any], path: str, value: Any) -> None:
    cursor = data
    parts = path.split(".")
    for part in parts[:-1]:
        # A scalar standing where the path expects a branch is replaced rather
        # than assigned into, which would raise.
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value
