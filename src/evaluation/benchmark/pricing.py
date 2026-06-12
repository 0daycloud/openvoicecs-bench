"""Pinned pricing manifests for reproducible frontier cost calculations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


DEFAULT_PRICING_MANIFEST_PATH = Path("data/openvoicecs/pricing_snapshot_v0.1.json")
SUPPORTED_COMPONENTS = {
    "asr",
    "llm",
    "tts",
    "speech_to_speech",
    "telephony",
    "transport",
}
PIPELINE_REQUIRED_COMPONENTS = {
    "cascaded": {"asr", "llm", "tts", "telephony", "transport"},
    "native_speech_to_speech": {"speech_to_speech", "telephony", "transport"},
}
COMPONENT_PRICING_KEYS = {
    "asr": {
        "asr_per_minute",
        "stt_per_minute",
        "asr_per_hour",
        "stt_per_hour",
    },
    "llm": {
        "input_per_mtok",
        "llm_input_per_mtok",
        "output_per_mtok",
        "llm_output_per_mtok",
        "cached_input_per_mtok",
    },
    "tts": {
        "tts_per_million_characters",
        "tts_per_million_chars",
        "tts_per_1k_characters",
        "tts_per_1k_chars",
        "tts_per_minute",
    },
    "speech_to_speech": {
        "speech_to_speech_per_minute",
        "s2s_per_minute",
        "realtime_per_minute",
        "input_audio_per_minute",
        "speech_input_per_minute",
        "output_audio_per_minute",
        "speech_output_per_minute",
        "input_audio_per_mtok",
        "audio_input_per_mtok",
        "output_audio_per_mtok",
        "audio_output_per_mtok",
    },
    "telephony": {
        "telephony_per_minute",
        "phone_per_minute",
    },
    "transport": {
        "transport_per_minute",
        "webrtc_per_minute",
    },
}


@dataclass(frozen=True)
class PricingIssue:
    """Structured pricing manifest validation issue."""

    scenario_id: str
    path: str
    message: str


def load_pricing_manifest(path: str | Path = DEFAULT_PRICING_MANIFEST_PATH) -> dict[str, Any]:
    """Load and validate a pinned pricing manifest."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    issues = validate_pricing_manifest(manifest)
    if issues:
        formatted = "\n".join(
            f"- {issue.scenario_id}::{issue.path}: {issue.message}"
            for issue in issues
        )
        raise ValueError(f"OpenVoiceCS pricing manifest validation failed:\n{formatted}")
    return manifest


def validate_pricing_manifest_file(
    path: str | Path = DEFAULT_PRICING_MANIFEST_PATH,
) -> list[PricingIssue]:
    """Validate a pricing manifest JSON file."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    return validate_pricing_manifest(manifest)


def validate_pricing_manifest(manifest: dict[str, Any]) -> list[PricingIssue]:
    """Return pricing manifest contract issues without mutating data."""
    issues: list[PricingIssue] = []
    if not isinstance(manifest, dict):
        return [PricingIssue("<pricing>", "<root>", "must be an object")]
    if not manifest.get("snapshot_date"):
        issues.append(PricingIssue("<pricing>", "snapshot_date", "missing required field"))
    if manifest.get("currency") != "USD":
        issues.append(PricingIssue("<pricing>", "currency", "must be USD"))

    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        issues.append(PricingIssue("<pricing>", "entries", "must be a non-empty list"))
        entries = []
    entry_ids = set()
    entry_components: dict[str, str] = {}
    components_seen = set()
    for index, entry in enumerate(entries):
        path = f"entries[{index}]"
        if not isinstance(entry, dict):
            issues.append(PricingIssue("<pricing>", path, "must be an object"))
            continue
        entry_id = entry.get("id")
        if not entry_id:
            issues.append(PricingIssue("<pricing>", f"{path}.id", "missing required field"))
        elif entry_id in entry_ids:
            issues.append(PricingIssue(str(entry_id), f"{path}.id", "duplicate entry id"))
        entry_ids.add(entry_id)
        component = entry.get("component")
        if component not in SUPPORTED_COMPONENTS:
            issues.append(PricingIssue(str(entry_id), f"{path}.component", "unsupported component"))
        else:
            components_seen.add(component)
            if entry_id:
                entry_components[str(entry_id)] = component
        for field in ("provider", "model_id", "pricing"):
            if field not in entry:
                issues.append(PricingIssue(str(entry_id), f"{path}.{field}", "missing required field"))
        pricing = entry.get("pricing")
        if not isinstance(pricing, dict) or not pricing:
            issues.append(PricingIssue(str(entry_id), f"{path}.pricing", "must be a non-empty object"))
        elif not any(_is_nonnegative_number(value) for value in pricing.values()):
            issues.append(PricingIssue(str(entry_id), f"{path}.pricing", "must contain numeric rates"))
        elif component in COMPONENT_PRICING_KEYS and not any(
            key in COMPONENT_PRICING_KEYS[component] and _is_nonnegative_number(value)
            for key, value in pricing.items()
        ):
            issues.append(
                PricingIssue(
                    str(entry_id),
                    f"{path}.pricing",
                    f"must include a numeric {component} pricing key",
                )
            )

    profiles = manifest.get("profiles", [])
    if not isinstance(profiles, list):
        issues.append(PricingIssue("<pricing>", "profiles", "must be a list"))
        profiles = []
    profile_ids = set()
    for index, profile in enumerate(profiles):
        path = f"profiles[{index}]"
        if not isinstance(profile, dict):
            issues.append(PricingIssue("<pricing>", path, "must be an object"))
            continue
        profile_id = profile.get("id")
        if not profile_id:
            issues.append(PricingIssue("<pricing>", f"{path}.id", "missing required field"))
        elif profile_id in profile_ids:
            issues.append(PricingIssue(str(profile_id), f"{path}.id", "duplicate profile id"))
        profile_ids.add(profile_id)
        components = profile.get("components")
        if not isinstance(components, dict):
            issues.append(PricingIssue(str(profile_id), f"{path}.components", "must be an object"))
            continue
        pipeline_type = profile.get("pipeline_type", "cascaded")
        if pipeline_type not in PIPELINE_REQUIRED_COMPONENTS:
            issues.append(PricingIssue(str(profile_id), f"{path}.pipeline_type", "unsupported pipeline type"))
            required_components = PIPELINE_REQUIRED_COMPONENTS["cascaded"]
        else:
            required_components = PIPELINE_REQUIRED_COMPONENTS[pipeline_type]
        missing_components = required_components - set(components)
        if missing_components:
            issues.append(
                PricingIssue(
                    str(profile_id),
                    f"{path}.components",
                    f"missing components: {', '.join(sorted(missing_components))}",
                )
            )
        for component, entry_id in components.items():
            if component not in SUPPORTED_COMPONENTS:
                issues.append(PricingIssue(str(profile_id), f"{path}.components.{component}", "unsupported component"))
            if entry_id not in entry_ids:
                issues.append(PricingIssue(str(profile_id), f"{path}.components.{component}", "unknown entry id"))
            elif component in SUPPORTED_COMPONENTS:
                entry_component = entry_components.get(str(entry_id))
                if entry_component and entry_component != component:
                    issues.append(
                        PricingIssue(
                            str(profile_id),
                            f"{path}.components.{component}",
                            f"entry component mismatch: expected {component}, got {entry_component}",
                        )
                    )

    required_entry_components = set()
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        pipeline_type = profile.get("pipeline_type", "cascaded")
        required_entry_components.update(
            PIPELINE_REQUIRED_COMPONENTS.get(
                pipeline_type,
                PIPELINE_REQUIRED_COMPONENTS["cascaded"],
            )
        )
    missing_entry_components = required_entry_components - components_seen
    if missing_entry_components:
        issues.append(
            PricingIssue(
                "<pricing>",
                "entries",
                f"missing component entries: {', '.join(sorted(missing_entry_components))}",
            )
        )
    return issues


def resolve_report_pricing(
    report: dict[str, Any],
    pricing_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve pricing rates for a report from embedded data or manifest profile."""
    metadata = report.get("model_metadata", {})
    embedded = report.get("pricing") or metadata.get("pricing") or {}
    if not pricing_manifest:
        return dict(embedded) if isinstance(embedded, dict) else {}

    resolved: dict[str, Any] = {
        "snapshot_date": pricing_manifest.get("snapshot_date"),
        "currency": pricing_manifest.get("currency", "USD"),
    }
    profile = _resolve_profile(metadata, pricing_manifest)
    if profile:
        entries = {entry["id"]: entry for entry in pricing_manifest.get("entries", [])}
        resolved["profile_id"] = profile.get("id")
        resolved["pipeline_type"] = profile.get("pipeline_type", "cascaded")
        resolved["component_entry_ids"] = dict(profile.get("components", {}))
        for entry_id in profile.get("components", {}).values():
            entry = entries.get(entry_id)
            if entry:
                resolved.update(entry.get("pricing", {}))
    if isinstance(embedded, dict):
        resolved.update(embedded)
    return resolved


def pricing_manifest_stats(manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize a pricing manifest for release audits."""
    if not manifest:
        return {"present": False, "num_entries": 0, "num_profiles": 0}
    entries = manifest.get("entries", [])
    profiles = manifest.get("profiles", [])
    comparable_profiles = [
        profile for profile in profiles
        if isinstance(profile, dict) and _is_comparable_pricing_profile(profile)
    ] if isinstance(profiles, list) else []
    return {
        "present": True,
        "snapshot_date": manifest.get("snapshot_date"),
        "currency": manifest.get("currency"),
        "num_entries": len(entries) if isinstance(entries, list) else 0,
        "num_profiles": len(profiles) if isinstance(profiles, list) else 0,
        "num_comparable_profiles": len(comparable_profiles),
        "components": sorted({
            entry.get("component")
            for entry in entries
            if isinstance(entry, dict) and entry.get("component")
        }),
    }


def _resolve_profile(
    metadata: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    profiles = manifest.get("profiles", [])
    if not isinstance(profiles, list):
        return None
    profile_id = (
        metadata.get("pricing_profile_id")
        or metadata.get("pricing_profile")
        or metadata.get("profile_id")
    )
    if profile_id:
        return next((profile for profile in profiles if profile.get("id") == profile_id), None)
    provider = metadata.get("provider")
    model_id = metadata.get("model_id")
    if provider and model_id:
        return next(
            (
                profile for profile in profiles
                if profile.get("provider") == provider and profile.get("model_id") == model_id
            ),
            None,
        )
    return profiles[0] if len(profiles) == 1 else None


def _is_comparable_pricing_profile(profile: dict[str, Any]) -> bool:
    provider = str(profile.get("provider") or "").strip().lower()
    profile_id = str(profile.get("id") or "").strip().lower()
    model_id = str(profile.get("model_id") or "").strip().lower()
    return bool(provider and model_id) and provider != "reference" and not profile_id.startswith("reference")


def _is_nonnegative_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value >= 0
