"""Provider-agnostic LLM wrapper (free-tier friendly).

Providers (set LLM_PROVIDER):
  gemini      Google AI Studio free tier       (key: GEMINI_API_KEY / GOOGLE_API_KEY)  [default]
  groq        Groq free tier                   (key: GROQ_API_KEY)
  openrouter  OpenRouter ':free' models        (key: OPENROUTER_API_KEY)
  ollama      local models, no key             (needs `ollama serve`)
  anthropic   Claude API (paid)                (key: ANTHROPIC_API_KEY)

All but `anthropic` use the OpenAI-compatible protocol, so one code path serves them.
An optional LLM_FALLBACK_PROVIDER is tried automatically if the primary fails
(rate limit, outage, bad output), which keeps a live demo alive.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    model: str
    key_envs: tuple[str, ...]
    base_url: str | None = None
    default_key: str | None = None
    native_anthropic: bool = False


# Model ids on free tiers change often: override with LLM_MODEL if one is retired.
PRESETS: dict[str, Preset] = {
    "gemini": Preset("gemini-flash-latest", ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
                     "https://generativelanguage.googleapis.com/v1beta/openai/"),
    "groq": Preset("llama-3.3-70b-versatile", ("GROQ_API_KEY",), "https://api.groq.com/openai/v1"),
    "openrouter": Preset("openrouter/free", ("OPENROUTER_API_KEY",), "https://openrouter.ai/api/v1"),
    "ollama": Preset("qwen2.5:7b", (), "http://localhost:11434/v1", default_key="ollama"),
    "anthropic": Preset("claude-sonnet-5-5", ("ANTHROPIC_API_KEY",), native_anthropic=True),
}


def _is_rate_limit(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 429 or "ratelimit" in type(exc).__name__.lower()


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError("no JSON object in model output")
        return json.loads(m.group(0))


# --------------------------------------------------------------------------- backends
class _OpenAIBackend:
    """Any OpenAI-compatible chat-completions endpoint."""

    def __init__(self, label: str, model: str, client):
        self.label, self.model, self.client = label, model, client

    def complete(self, *, system: str, user: str, max_tokens: int) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (resp.choices[0].message.content or "").strip()

    def structured(self, *, system: str, user: str, schema: dict, name: str,
                   description: str, max_tokens: int) -> dict:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:  # 1) forced function call (best schema adherence)
            resp = self.client.chat.completions.create(
                model=self.model, max_tokens=max_tokens, messages=messages,
                tools=[{"type": "function", "function": {"name": name, "description": description,
                                                          "parameters": schema}}],
                tool_choice={"type": "function", "function": {"name": name}},
            )
            msg = resp.choices[0].message
            if getattr(msg, "tool_calls", None):
                return json.loads(msg.tool_calls[0].function.arguments)
            if msg.content:
                return _extract_json(msg.content)
        except Exception as exc:
            if _is_rate_limit(exc):
                raise
        # 2) JSON-mode fallback for models without tool calling (e.g. small local models)
        messages[0]["content"] += ("\n\nRespond with ONLY a JSON object matching this JSON schema:\n"
                                   + json.dumps(schema))
        try:
            resp = self.client.chat.completions.create(
                model=self.model, max_tokens=max_tokens, messages=messages,
                response_format={"type": "json_object"})
        except Exception as exc:
            if _is_rate_limit(exc):
                raise
            resp = self.client.chat.completions.create(
                model=self.model, max_tokens=max_tokens, messages=messages)
        return _extract_json(resp.choices[0].message.content or "")


class _AnthropicBackend:
    def __init__(self, label: str, model: str, client):
        self.label, self.model, self.client = label, model, client

    def complete(self, *, system: str, user: str, max_tokens: int) -> str:
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def structured(self, *, system: str, user: str, schema: dict, name: str,
                   description: str, max_tokens: int) -> dict:
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
            tools=[{"name": name, "description": description, "input_schema": schema}],
            tool_choice={"type": "tool", "name": name})
        for block in resp.content:
            if block.type == "tool_use":
                return dict(block.input)
        raise RuntimeError("Model returned no structured output.")


def _build(provider: str, *, model: str | None, base_url: str | None, api_key: str | None, required: bool):
    preset = PRESETS.get(provider)
    if preset is None:
        raise ValueError(f"Unknown LLM provider {provider!r}. Choose from: {', '.join(PRESETS)}")
    key = api_key or next((os.getenv(k) for k in preset.key_envs if os.getenv(k)), None) or preset.default_key
    if not key:
        if required:
            raise RuntimeError(f"No API key for provider '{provider}': set {preset.key_envs[0]} in .env")
        return None
    model = model or preset.model
    label = f"{provider}:{model}"
    if preset.native_anthropic:
        import anthropic

        return _AnthropicBackend(label, model, anthropic.Anthropic(api_key=key))
    from openai import OpenAI

    return _OpenAIBackend(label, model, OpenAI(api_key=key, base_url=base_url or preset.base_url))


# --------------------------------------------------------------------------- public class
class LLM:
    """Tries each backend in order; retries once on HTTP 429 before moving on."""

    def __init__(self, backends: list):
        if not backends:
            raise ValueError("LLM needs at least one backend")
        self.backends = backends
        self.last_used: str | None = None

    @classmethod
    def from_env(cls) -> "LLM":
        primary = _build(os.getenv("LLM_PROVIDER", "gemini").strip().lower(),
                         model=os.getenv("LLM_MODEL") or None, base_url=os.getenv("LLM_BASE_URL") or None,
                         api_key=os.getenv("LLM_API_KEY") or None, required=True)
        backends = [primary]
        fb = os.getenv("LLM_FALLBACK_PROVIDER", "").strip().lower()
        if fb:
            extra = _build(fb, model=os.getenv("LLM_FALLBACK_MODEL") or None, base_url=None,
                           api_key=None, required=False)
            if extra:
                backends.append(extra)
        return cls(backends)

    def _run(self, method: str, **kwargs):
        errors = []
        for backend in self.backends:
            for attempt in (0, 1):
                try:
                    out = getattr(backend, method)(**kwargs)
                    self.last_used = backend.label
                    return out
                except Exception as exc:
                    if attempt == 0 and _is_rate_limit(exc):
                        time.sleep(2)
                        continue
                    errors.append(f"{backend.label}: {exc}")
                    break
        raise RuntimeError("All LLM backends failed -> " + " | ".join(errors))

    def complete(self, *, system: str, user: str, max_tokens: int = 2000) -> str:
        return self._run("complete", system=system, user=user, max_tokens=max_tokens)

    def structured(self, *, system: str, user: str, schema: dict, name: str,
                   description: str = "Return the structured result.", max_tokens: int = 1500) -> dict:
        return self._run("structured", system=system, user=user, schema=schema, name=name,
                         description=description, max_tokens=max_tokens)