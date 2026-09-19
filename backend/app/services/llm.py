"""Provider-agnostic LLM client interfaces and implementations.

Supports:
- OpenAI API (GPT-4o, GPT-4o-mini, etc.)
- Anthropic Claude API (Claude 3.5 Sonnet, Haiku, etc.)
- Ollama (local open-source models via REST API)
- MockLLMClient for deterministic unit/integration testing
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract interface for large language model text generation."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the active model name/identifier."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Synchronously generate text completion from prompt.

        Args:
            prompt: User prompt or main task instructions.
            system_prompt: Optional system persona and grounding instructions.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in response.

        Returns:
            str: Generated text response.
        """

    @abstractmethod
    async def agenerate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Asynchronously generate text completion from prompt.

        Args:
            prompt: User prompt or main task instructions.
            system_prompt: Optional system persona and grounding instructions.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in response.

        Returns:
            str: Generated text response.
        """


class OpenAILLMClient(BaseLLMClient):
    """OpenAI API client implementation."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.OPENAI_MODEL
        self.base_url = base_url or settings.OPENAI_BASE_URL
        self._sync_client: Any = None
        self._async_client: Any = None

    @property
    def model_name(self) -> str:
        return self.model

    def _get_sync_client(self) -> Any:
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Please provide a key in settings or pass api_key."
            )
        if self._sync_client is None:
            from openai import OpenAI

            self._sync_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._sync_client

    def _get_async_client(self) -> Any:
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Please provide a key in settings or pass api_key."
            )
        if self._async_client is None:
            from openai import AsyncOpenAI

            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._async_client

    def _build_messages(
        self, prompt: str, system_prompt: str | None
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        client = self._get_sync_client()
        messages = self._build_messages(prompt, system_prompt)
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
        tokens = max_tokens or settings.LLM_MAX_TOKENS

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp,
            max_tokens=tokens,
        )
        return str(response.choices[0].message.content or "")

    async def agenerate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        client = self._get_async_client()
        messages = self._build_messages(prompt, system_prompt)
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
        tokens = max_tokens or settings.LLM_MAX_TOKENS

        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp,
            max_tokens=tokens,
        )
        return str(response.choices[0].message.content or "")


class ClaudeLLMClient(BaseLLMClient):
    """Anthropic Claude API client implementation."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key or settings.ANTHROPIC_API_KEY
        self.model = model or settings.ANTHROPIC_MODEL
        self.base_url = base_url or settings.ANTHROPIC_BASE_URL
        self._sync_client: Any = None
        self._async_client: Any = None

    @property
    def model_name(self) -> str:
        return self.model

    def _get_sync_client(self) -> Any:
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Please provide a key in settings or pass api_key."
            )
        if self._sync_client is None:
            from anthropic import Anthropic

            self._sync_client = Anthropic(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._sync_client

    def _get_async_client(self) -> Any:
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Please provide a key in settings or pass api_key."
            )
        if self._async_client is None:
            from anthropic import AsyncAnthropic

            self._async_client = AsyncAnthropic(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._async_client

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        client = self._get_sync_client()
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
        tokens = max_tokens or settings.LLM_MAX_TOKENS

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": tokens,
            "temperature": temp,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = client.messages.create(**kwargs)
        blocks = response.content
        if blocks and hasattr(blocks[0], "text"):
            return str(blocks[0].text)
        return ""

    async def agenerate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        client = self._get_async_client()
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
        tokens = max_tokens or settings.LLM_MAX_TOKENS

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": tokens,
            "temperature": temp,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await client.messages.create(**kwargs)
        blocks = response.content
        if blocks and hasattr(blocks[0], "text"):
            return str(blocks[0].text)
        return ""


class OllamaLLMClient(BaseLLMClient):
    """Local Ollama REST API client implementation."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout

    @property
    def model_name(self) -> str:
        return self.model

    def _build_payload(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        options: dict[str, Any] = {
            "temperature": (
                temperature if temperature is not None else settings.LLM_TEMPERATURE
            ),
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        return {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        import httpx

        url = f"{self.base_url}/api/chat"
        payload = self._build_payload(prompt, system_prompt, temperature, max_tokens)

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return str(data.get("message", {}).get("content", ""))

    async def agenerate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        import httpx

        url = f"{self.base_url}/api/chat"
        payload = self._build_payload(prompt, system_prompt, temperature, max_tokens)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return str(data.get("message", {}).get("content", ""))


class MockLLMClient(BaseLLMClient):
    """Deterministic mock client for testing and local simulation without API keys."""

    def __init__(
        self,
        default_response: str = "Mock answer grounded in context [1].",
        responses: list[str] | None = None,
        model: str = "mock-model",
    ) -> None:
        self._default_response = default_response
        self._responses = list(responses) if responses else []
        self._model = model
        self.call_history: list[dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return self._model

    def _record_and_get_response(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> str:
        self.call_history.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if self._responses:
            return self._responses.pop(0)
        return self._default_response

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return self._record_and_get_response(
            prompt, system_prompt, temperature, max_tokens
        )

    async def agenerate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return self._record_and_get_response(
            prompt, system_prompt, temperature, max_tokens
        )


def get_llm_client(provider: str | None = None, **kwargs: Any) -> BaseLLMClient:
    """Factory function resolving and instantiating the configured LLM client.

    Args:
        provider: Provider identifier ('openai', 'anthropic' / 'claude', 'ollama', 'mock').
            If None, reads from settings.LLM_PROVIDER.
        **kwargs: Optional keyword arguments passed directly to the client constructor.

    Returns:
        BaseLLMClient: Configured LLM client instance.

    Raises:
        ValueError: If provider is unrecognized.
    """
    selected = (provider or settings.LLM_PROVIDER).strip().lower()

    if selected == "openai":
        return OpenAILLMClient(**kwargs)
    elif selected in ("anthropic", "claude"):
        return ClaudeLLMClient(**kwargs)
    elif selected == "ollama":
        return OllamaLLMClient(**kwargs)
    elif selected == "mock":
        return MockLLMClient(**kwargs)
    else:
        raise ValueError(
            f"Unsupported LLM provider '{selected}'. "
            "Supported options: 'openai', 'anthropic' (or 'claude'), 'ollama', 'mock'."
        )
