"""vLLM client for self-hosted LLM inference."""

import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

LOGGER = logging.getLogger(__name__)


@dataclass
class VLLMConfig:
    """Configuration for vLLM server connection."""

    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"
    api_key: str = "EMPTY"
    timeout: float = 180.0

    @classmethod
    def from_env(cls) -> "VLLMConfig":
        return cls(
            base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
            model=os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"),
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            timeout=float(os.environ.get("VLLM_TIMEOUT", "180.0")),
        )


@dataclass
class CompletionResponse:
    """Response from vLLM completion."""

    content: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: str = "stop"


class VLLMClient:
    """Client for vLLM OpenAI-compatible API."""

    def __init__(self, config: Optional[VLLMConfig] = None):
        self.config = config or VLLMConfig.from_env()
        self._client = httpx.Client(timeout=self.config.timeout)
        self._async_client: Optional[httpx.AsyncClient] = None

    async def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create the persistent async client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.config.timeout)
        return self._async_client

    def completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> CompletionResponse:
        """Synchronous chat completion."""
        response = self._client.post(
            f"{self.config.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            json={
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 1024,
                "stop": stop,
            },
        )
        response.raise_for_status()
        data = response.json()

        return CompletionResponse(
            content=data["choices"][0]["message"]["content"],
            usage=data.get("usage"),
            finish_reason=data["choices"][0].get("finish_reason", "stop"),
        )

    async def acompletion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> CompletionResponse:
        """Asynchronous chat completion with connection reuse."""
        client = await self._get_async_client()
        response = await client.post(
            f"{self.config.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            json={
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 1024,
                "stop": stop,
            },
        )
        response.raise_for_status()
        data = response.json()

        return CompletionResponse(
            content=data["choices"][0]["message"]["content"],
            usage=data.get("usage"),
            finish_reason=data["choices"][0].get("finish_reason", "stop"),
        )

    async def aclose(self):
        """Close the async HTTP client."""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None

    def close(self):
        """Close the sync HTTP client."""
        self._client.close()


# Module-level singleton for convenience
_default_client: Optional[VLLMClient] = None


def get_client() -> VLLMClient:
    """Get or create the default vLLM client."""
    global _default_client
    if _default_client is None:
        _default_client = VLLMClient()
    return _default_client


def completion(
    messages: List[Dict[str, str]],
    model: str = None,  # Ignored, uses configured model
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    timeout: float = None,  # Ignored, uses configured timeout
    **kwargs,
) -> CompletionResponse:
    """Convenience function matching litellm.completion signature."""
    return get_client().completion(messages, temperature, max_tokens)


__all__ = [
    "VLLMConfig",
    "VLLMClient",
    "CompletionResponse",
    "get_client",
    "completion",
]
