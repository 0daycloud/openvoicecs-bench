"""Provider adapters for running OpenVoiceCS-Bench against hosted LLMs.

The benchmark evaluates a trace, not free-form chat text. Each provider is
therefore asked to produce one small JSON trace containing messages, tool
calls, policy events, and optional claims. The provider-specific layer is kept
thin so official runs can pin model IDs and pricing externally.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

PIPELINE_PROVIDERS = {
    "openai",
    "google",
    "anthropic",
    "alibaba",
    "kimi",
    "minimax",
    "deepseek",
    "xai",
    "openrouter",
}

OPENAI_COMPATIBLE_BASE_URLS = {
    "alibaba": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "kimi": "https://api.moonshot.ai/v1",
    "minimax": "https://api.minimax.io/v1",
    "deepseek": "https://api.deepseek.com",
    "xai": "https://api.x.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

PROVIDER_ENV_KEYS = {
    "openai": ("OPENAI_API_KEY",),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "alibaba": ("DASHSCOPE_API_KEY", "ALIBABA_API_KEY", "ALIBABA_CLOUD_API_KEY"),
    "kimi": ("MOONSHOT_API_KEY", "KIMI_API_KEY"),
    "minimax": ("MINIMAX_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "xai": ("XAI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY", "OPENROUTER_KEY", "OPEN_ROUTER_API_KEY"),
}

DEFAULT_MODEL_IDS = {
    "openai": "gpt-5.4-mini",
    "google": "gemini-2.5-flash",
    "anthropic": "claude-sonnet-4-6",
    "alibaba": "qwen-max",
    "kimi": "kimi-k2.5",
    "minimax": "MiniMax-M1",
    "deepseek": "deepseek-chat",
    "xai": "grok-4",
    "openrouter": "openai/gpt-5.4-mini",
}

DEFAULT_MAX_OUTPUT_TOKENS = 700
DEFAULT_TEMPERATURE = 0.1
REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh"}


@dataclass(frozen=True)
class ProviderSpec:
    """Configuration for one hosted model adapter."""

    provider: str
    model_id: str
    display_name: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    reasoning_effort: str | None = None
    pricing: dict[str, Any] | None = None

    def metadata(self, *, input_modality: str, pricing_snapshot_date: str | None = None) -> dict[str, Any]:
        return {
            "display_name": self.display_name or self.model_id,
            "provider": self.provider,
            "model_id": self.model_id,
            "input_modality": input_modality,
            "pipeline_type": "cascaded",
            "adapter": "openvoicecs-provider-adapter-v0.1",
            "reasoning_effort": self.reasoning_effort,
            "pricing": self.pricing or {},
            "pricing_snapshot_date": pricing_snapshot_date,
        }


def build_provider_agent(spec: ProviderSpec) -> Callable[[dict[str, Any], int], dict[str, Any]]:
    """Create a benchmark agent callable for a provider spec."""
    provider = _normalize_provider(spec.provider)
    spec = ProviderSpec(
        provider=provider,
        model_id=spec.model_id or DEFAULT_MODEL_IDS[provider],
        display_name=spec.display_name,
        api_key=spec.api_key,
        base_url=spec.base_url,
        max_output_tokens=spec.max_output_tokens,
        temperature=spec.temperature,
        reasoning_effort=spec.reasoning_effort,
        pricing=spec.pricing,
    )
    if provider == "anthropic":
        return _build_anthropic_agent(spec)
    if provider == "google":
        return _build_google_agent(spec)
    if provider in {"openai", "alibaba", "kimi", "minimax", "deepseek", "xai", "openrouter"}:
        return _build_openai_compatible_agent(spec)
    raise ValueError(f"unsupported provider: {provider}")


def build_provider_spec(
    provider: str,
    *,
    model_id: str | None = None,
    display_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    reasoning_effort: str | None = None,
    pricing: dict[str, Any] | None = None,
) -> ProviderSpec:
    """Normalize CLI/user provider inputs into a provider spec."""
    provider = _normalize_provider(provider)
    if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f"reasoning_effort must be one of: {', '.join(sorted(REASONING_EFFORTS))}")
    return ProviderSpec(
        provider=provider,
        model_id=model_id or DEFAULT_MODEL_IDS[provider],
        display_name=display_name,
        api_key=api_key,
        base_url=base_url,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        pricing=pricing,
    )


def provider_metadata(
    spec: ProviderSpec,
    *,
    input_modality: str,
    pricing_profile_id: str | None = None,
    pricing_snapshot_date: str | None = None,
) -> dict[str, Any]:
    """Return report metadata for a provider adapter run."""
    metadata = spec.metadata(
        input_modality=input_modality,
        pricing_snapshot_date=pricing_snapshot_date,
    )
    metadata["pricing_profile_id"] = pricing_profile_id
    return metadata


def parse_provider_response_text(text: str) -> dict[str, Any]:
    """Parse and normalize a provider JSON trace response."""
    payload = _extract_json_object(text)
    trace = json.loads(payload)
    if not isinstance(trace, dict):
        raise ValueError("provider response JSON must be an object")

    messages = trace.get("messages") or []
    if isinstance(messages, str):
        messages = [{"role": "agent", "text": messages}]
    if not messages and isinstance(trace.get("response"), str):
        messages = [{"role": "agent", "text": trace["response"]}]

    experience_judgment = trace.get("experience_judgment") or trace.get("conversation_experience")
    if (
        isinstance(experience_judgment, dict)
        and "overall" in experience_judgment
        and "score" not in experience_judgment
        and "overall_score" not in experience_judgment
    ):
        experience_judgment = {**experience_judgment, "score": experience_judgment["overall"]}

    return {
        "messages": messages,
        "tool_calls": trace.get("tool_calls") or [],
        "events": trace.get("events") or [],
        "claims": trace.get("claims") or trace.get("grounding_claims") or [],
        "latency": trace.get("latency"),
        "latency_ms": trace.get("latency_ms"),
        "experience_judgment": experience_judgment,
    }


def build_trace_prompt(scenario: dict[str, Any], trial_index: int) -> tuple[str, str]:
    """Build the provider-independent system and user prompts."""
    customer_text = _scenario_customer_text(scenario)
    tool_specs = [
        {
            "name": tool.get("name"),
            "description": tool.get("description") or _tool_description(tool),
            "required_arguments": tool.get("required_arguments") or {},
            "state_effects": _tool_state_effects(tool),
        }
        for tool in scenario.get("tools", [])
    ]
    oracle = scenario.get("oracle", {})
    required_events = _unique_strings(
        list(scenario.get("policy", {}).get("required_events") or [])
        + list(oracle.get("required_events") or [])
        + list(oracle.get("privacy", {}).get("required_events") or [])
        + list(oracle.get("auth", {}).get("required_events") or [])
    )
    forbidden_events = _unique_strings(
        list(oracle.get("forbidden_events") or [])
        + list(oracle.get("privacy", {}).get("forbidden_events") or [])
        + list(oracle.get("auth", {}).get("forbidden_events") or [])
    )
    scenario_view = {
        "id": scenario.get("id"),
        "domain": scenario.get("domain"),
        "track": scenario.get("track"),
        "difficulty": scenario.get("difficulty"),
        "customer_goal": scenario.get("customer_goal"),
        "customer_profile": scenario.get("customer_profile", {}),
        "customer_utterance": customer_text,
        "initial_state": scenario.get("initial_state", {}),
        "policy": scenario.get("policy", {}),
        "available_tools": tool_specs,
        "required_events": required_events,
        "forbidden_events": forbidden_events,
        "forbidden_tool_calls": oracle.get("forbidden_tool_calls", []),
        "experience": scenario.get("experience", {}),
        "audio_variant": _summarize_audio_variant(scenario.get("audio_variant")),
        "trial_index": trial_index,
    }

    system = (
        "You are an evaluated customer-service voice agent. Produce only valid JSON. "
        "Do not wrap it in Markdown. The benchmark will replay your tool_calls; do not "
        "invent tools or arguments that are not listed. Follow policy and avoid "
        "forbidden events/tools. Keep the final agent message concise and natural."
    )
    user = (
        "Return exactly this JSON object shape:\n"
        "{\n"
        '  "messages": [{"role": "agent", "text": "..."}],\n'
        '  "tool_calls": [{"name": "tool_name", "arguments": {}}],\n'
        '  "events": ["policy_event"],\n'
        '  "claims": [{"text": "short factual claim", "supported": true}],\n'
        '  "experience_judgment": {"score": 1, "notes": "optional"}\n'
        "}\n\n"
        "Scenario:\n"
        f"{json.dumps(scenario_view, ensure_ascii=True, indent=2)}"
    )
    return system, user


def estimate_cost_usd(usage: dict[str, Any], pricing: dict[str, Any] | None) -> float | None:
    """Estimate text-token API cost from usage and per-million-token pricing."""
    if not pricing:
        return None
    input_price = pricing.get("input_per_mtok")
    output_price = pricing.get("output_per_mtok")
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if input_price is None or output_price is None:
        return None
    if not isinstance(input_tokens, (int, float)) or not isinstance(output_tokens, (int, float)):
        return None
    return round(
        (float(input_tokens) / 1_000_000 * float(input_price))
        + (float(output_tokens) / 1_000_000 * float(output_price)),
        8,
    )


def _tool_description(tool: dict[str, Any]) -> str:
    """Return a concise provider-facing description for a scenario tool."""
    name = str(tool.get("name") or "tool")
    state_effects = _tool_state_effects(tool)
    if not state_effects:
        return f"Call {name} when the scenario policy requires this operation."
    effect_text = "; ".join(
        f"sets {effect['path']} to {json.dumps(effect['value'], ensure_ascii=True)}"
        for effect in state_effects
    )
    return f"Call {name} when the scenario policy requires this operation; it {effect_text}."


def _tool_state_effects(tool: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose deterministic replay effects in provider prompts."""
    effects = []
    for update in tool.get("state_updates") or []:
        if not isinstance(update, dict):
            continue
        effects.append({
            "path": update.get("path"),
            "value": update.get("value"),
        })
    return effects


def load_workspace_env(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE pairs without overriding existing environment values."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def get_provider_api_key(provider: str, explicit: str | None = None) -> str | None:
    """Return an API key from explicit value or known environment names."""
    if explicit:
        return explicit
    for key_name in PROVIDER_ENV_KEYS[_normalize_provider(provider)]:
        value = os.getenv(key_name)
        if value:
            return value
    return None


def _build_openai_compatible_agent(spec: ProviderSpec) -> Callable[[dict[str, Any], int], dict[str, Any]]:
    from openai import OpenAI

    api_key = get_provider_api_key(spec.provider, spec.api_key)
    if not api_key:
        raise ValueError(
            f"{'/'.join(PROVIDER_ENV_KEYS[spec.provider])} is required for provider={spec.provider}"
        )
    kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = spec.base_url or OPENAI_COMPATIBLE_BASE_URLS.get(spec.provider)
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)

    def run(scenario: dict[str, Any], trial_index: int) -> dict[str, Any]:
        system, user = build_trace_prompt(scenario, trial_index)
        started = time.perf_counter()
        request: dict[str, Any] = {
            "model": spec.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": spec.temperature,
        }
        if spec.provider == "openai":
            request["max_completion_tokens"] = spec.max_output_tokens
        else:
            request["max_tokens"] = spec.max_output_tokens
        if spec.reasoning_effort is not None:
            request["reasoning_effort"] = spec.reasoning_effort
        response = client.chat.completions.create(**request)
        latency_ms = (time.perf_counter() - started) * 1000
        text = _extract_openai_message_text(response.choices[0].message)
        trace = parse_provider_response_text(text)
        usage = _extract_openai_usage(response)
        trace["usage"] = usage
        trace["cost_usd"] = estimate_cost_usd(usage, spec.pricing)
        trace.setdefault("latency_ms", round(latency_ms, 3))
        return trace

    return run


def _build_anthropic_agent(spec: ProviderSpec) -> Callable[[dict[str, Any], int], dict[str, Any]]:
    import anthropic

    api_key = get_provider_api_key(spec.provider, spec.api_key)
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def run(scenario: dict[str, Any], trial_index: int) -> dict[str, Any]:
        system, user = build_trace_prompt(scenario, trial_index)
        started = time.perf_counter()
        message = client.messages.create(
            model=spec.model_id,
            max_tokens=spec.max_output_tokens,
            temperature=spec.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        latency_ms = (time.perf_counter() - started) * 1000
        text = _extract_anthropic_text(message)
        trace = parse_provider_response_text(text)
        usage = _extract_anthropic_usage(message)
        trace["usage"] = usage
        trace["cost_usd"] = estimate_cost_usd(usage, spec.pricing)
        trace.setdefault("latency_ms", round(latency_ms, 3))
        return trace

    return run


def _build_google_agent(spec: ProviderSpec) -> Callable[[dict[str, Any], int], dict[str, Any]]:
    from google import genai

    api_key = get_provider_api_key(spec.provider, spec.api_key)
    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def run(scenario: dict[str, Any], trial_index: int) -> dict[str, Any]:
        system, user = build_trace_prompt(scenario, trial_index)
        started = time.perf_counter()
        response = client.models.generate_content(
            model=spec.model_id,
            contents=user,
            config={
                "system_instruction": system,
                "temperature": spec.temperature,
                "max_output_tokens": spec.max_output_tokens,
            },
        )
        latency_ms = (time.perf_counter() - started) * 1000
        trace = parse_provider_response_text(response.text or "")
        usage = _extract_google_usage(response)
        trace["usage"] = usage
        trace["cost_usd"] = estimate_cost_usd(usage, spec.pricing)
        trace.setdefault("latency_ms", round(latency_ms, 3))
        return trace

    return run


def _extract_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("provider response did not contain a JSON object")
    candidate = text[start : end + 1]
    json.loads(candidate)
    return candidate


def _extract_openai_message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str) and content:
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        if parts:
            return "\n".join(parts)
    raise ValueError("no text content in provider response")


def _extract_anthropic_text(message: Any) -> str:
    parts = []
    for item in getattr(message, "content", []) or []:
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
        elif isinstance(item, dict) and item.get("text"):
            parts.append(str(item["text"]))
    if not parts:
        raise ValueError("no text content in Anthropic response")
    return "\n".join(parts)


def _extract_openai_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if prompt_tokens is None and isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
    }


def _extract_anthropic_usage(message: Any) -> dict[str, int]:
    usage = getattr(message, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "cached_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }


def _extract_google_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
    }


def _scenario_customer_text(scenario: dict[str, Any]) -> str:
    audio_variant = scenario.get("audio_variant") or {}
    if isinstance(audio_variant, dict) and audio_variant.get("transcript"):
        return str(audio_variant["transcript"])
    utterance = scenario.get("user_utterance")
    if utterance:
        return str(utterance)
    conversation = scenario.get("conversation") or []
    if isinstance(conversation, list):
        parts = [
            str(turn.get("text", ""))
            for turn in conversation
            if isinstance(turn, dict) and turn.get("role") in {"customer", "user", "patient"}
        ]
        if parts:
            return "\n".join(parts)
    return ""


def _summarize_audio_variant(variant: Any) -> dict[str, Any] | None:
    if not isinstance(variant, dict):
        return None
    return {
        "id": variant.get("id"),
        "track": variant.get("track"),
        "transcript": variant.get("transcript"),
        "perturbations": variant.get("perturbations", []),
        "audio": variant.get("audio", {}),
    }


def _unique_strings(values: list[Any]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if not isinstance(value, str) or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower().replace("_", "-")
    aliases = {
        "dashscope": "alibaba",
        "aliyun": "alibaba",
        "alibaba-cloud": "alibaba",
        "moonshot": "kimi",
        "moonshotai": "kimi",
        "mini-max": "minimax",
        "grok": "xai",
        "open-router": "openrouter",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in PIPELINE_PROVIDERS:
        raise ValueError(f"provider must be one of: {', '.join(sorted(PIPELINE_PROVIDERS))}")
    return normalized
