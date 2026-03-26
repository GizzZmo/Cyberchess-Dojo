"""
LLM Adapter layer for Cyberchess-Dojo.

Provides a unified interface for different LLM providers so that agents and
the orchestrator can work with any supported model without modification.

The adapters expose the same ``generate_content(prompt)`` API that
``google.generativeai.GenerativeModel`` uses, so all existing agent and
orchestrator code works with zero changes — Python's duck-typing does the rest.

Supported providers
-------------------
- ``gemini``  : Google Gemini (default)
- ``openai``  : OpenAI GPT-4o / GPT-4o-mini / etc.
- ``claude``  : Anthropic Claude 3+

Usage::

    from llm_adapter import create_adapter

    # Gemini (default, backward-compatible)
    adapter = create_adapter("gemini")

    # OpenAI GPT-4o
    adapter = create_adapter("openai", model_name="gpt-4o")

    # Anthropic Claude
    adapter = create_adapter("claude", model_name="claude-3-5-sonnet-20241022")

    # Agents / orchestrator use adapter just like a GenerativeModel:
    response = adapter.generate_content("What is 1+1?")
    print(response.text)   # "2"
"""

import os
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Minimal response wrapper
# ---------------------------------------------------------------------------

class _LLMResponse:
    """Thin wrapper exposing a ``.text`` attribute — mirrors the Gemini SDK response."""

    def __init__(self, text: str):
        self.text = text


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseLLMAdapter(ABC):
    """Abstract base class for LLM adapters."""

    @abstractmethod
    def generate_content(self, prompt: str) -> _LLMResponse:
        """Generate text from the model and return a response with a ``.text`` attribute."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The resolved model identifier string."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------

class GeminiAdapter(BaseLLMAdapter):
    """
    Adapter for Google Gemini models.

    Wraps ``google.generativeai.GenerativeModel``.  The underlying model
    object is also accessible via ``adapter.native_model`` for any code that
    still needs it directly.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: str = None):
        import google.generativeai as genai

        key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        genai.configure(api_key=key)
        self._model_name = model_name
        self.native_model = genai.GenerativeModel(model_name)

    def generate_content(self, prompt: str) -> _LLMResponse:
        response = self.native_model.generate_content(prompt)
        return _LLMResponse(response.text)

    @property
    def model_name(self) -> str:
        return self._model_name


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIAdapter(BaseLLMAdapter):
    """
    Adapter for OpenAI chat models (GPT-4o, GPT-4o-mini, o1-preview, etc.).

    Requires the ``openai`` package and an ``OPENAI_API_KEY`` environment variable
    (or pass ``api_key`` explicitly).
    """

    def __init__(self, model_name: str = "gpt-4o", api_key: str = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required to use the OpenAI provider. "
                "Install it with: pip install openai"
            ) from exc

        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = OpenAI(api_key=key)
        self._model_name = model_name

    def generate_content(self, prompt: str) -> _LLMResponse:
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content or ""
        return _LLMResponse(text)

    @property
    def model_name(self) -> str:
        return self._model_name


# ---------------------------------------------------------------------------
# Anthropic Claude
# ---------------------------------------------------------------------------

class ClaudeAdapter(BaseLLMAdapter):
    """
    Adapter for Anthropic Claude models (claude-3-5-sonnet, claude-3-opus, etc.).

    Requires the ``anthropic`` package and an ``ANTHROPIC_API_KEY`` environment
    variable (or pass ``api_key`` explicitly).
    """

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", api_key: str = None):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required to use the Claude provider. "
                "Install it with: pip install anthropic"
            ) from exc

        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = anthropic.Anthropic(api_key=key)
        self._model_name = model_name

    def generate_content(self, prompt: str) -> _LLMResponse:
        message = self._client.messages.create(
            model=self._model_name,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text if message.content else ""
        return _LLMResponse(text)

    @property
    def model_name(self) -> str:
        return self._model_name


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

#: Default model names per provider.
_PROVIDER_DEFAULTS: dict[str, str] = {
    "gemini": "gemini-1.5-flash",
    "openai": "gpt-4o",
    "claude": "claude-3-5-sonnet-20241022",
}


def create_adapter(
    provider: str,
    model_name: str = None,
    api_key: str = None,
) -> BaseLLMAdapter:
    """
    Factory function — create the appropriate ``BaseLLMAdapter`` for a provider.

    Args:
        provider:   One of ``'gemini'``, ``'openai'``, ``'claude'``.
        model_name: Model identifier string.  Uses the provider default if ``None``.
        api_key:    API key.  Falls back to the provider's environment variable if ``None``.

    Returns:
        A ready-to-use ``BaseLLMAdapter`` instance.

    Raises:
        ValueError: If an unsupported provider name is supplied.
    """
    provider = provider.lower()
    if provider not in _PROVIDER_DEFAULTS:
        supported = ", ".join(f"'{p}'" for p in _PROVIDER_DEFAULTS)
        raise ValueError(f"Unknown LLM provider '{provider}'. Supported providers: {supported}")

    resolved_model = model_name or _PROVIDER_DEFAULTS[provider]

    if provider == "gemini":
        return GeminiAdapter(model_name=resolved_model, api_key=api_key)
    if provider == "openai":
        return OpenAIAdapter(model_name=resolved_model, api_key=api_key)
    # provider == "claude"
    return ClaudeAdapter(model_name=resolved_model, api_key=api_key)
