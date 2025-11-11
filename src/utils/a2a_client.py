"""A2A client utilities for inter-agent communication."""

import asyncio
import uuid
from typing import AsyncGenerator, Dict, Optional

import httpx
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import (
    AgentCard,
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    SendStreamingMessageRequest,
    SendStreamingMessageSuccessResponse,
    TextPart,
)


class A2AMessenger:
    """Utility wrapper for A2A client operations with caching."""

    def __init__(self, timeout: float = 120.0):
        """Initialize messenger with HTTP client.

        Args:
            timeout: HTTP request timeout in seconds
        """
        self._httpx_client = httpx.AsyncClient(timeout=timeout)
        self._card_cache: Dict[str, AgentCard] = {}
        self._client_cache: Dict[str, A2AClient] = {}

    async def _resolve_card(self, base_url: str) -> AgentCard:
        """Resolve agent card, using cache if available."""
        if base_url in self._card_cache:
            return self._card_cache[base_url]

        resolver = A2ACardResolver(httpx_client=self._httpx_client, base_url=base_url)
        card = await resolver.get_agent_card()
        if card is None:
            raise RuntimeError(f"Unable to resolve agent card from {base_url}")
        self._card_cache[base_url] = card
        return card

    async def _client(self, base_url: str) -> A2AClient:
        """Get or create A2A client for base URL."""
        if base_url in self._client_cache:
            return self._client_cache[base_url]

        card = await self._resolve_card(base_url)
        client = A2AClient(httpx_client=self._httpx_client, agent_card=card)
        self._client_cache[base_url] = client
        return client

    async def send_text(
        self,
        base_url: str,
        content: str,
        *,
        context_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Message:
        """Send a text message to an agent.

        Args:
            base_url: Base URL of target agent
            content: Message content
            context_id: Optional context ID for conversation continuity
            task_id: Optional task ID

        Returns:
            Response message from agent
        """
        client = await self._client(base_url)
        message = Message(
            role=Role.user,
            parts=[Part(TextPart(text=content))],
            message_id=uuid.uuid4().hex,
            task_id=task_id,
            context_id=context_id,
        )
        params = MessageSendParams(message=message)
        request = SendMessageRequest(id=uuid.uuid4().hex, params=params)
        response: SendMessageResponse = await client.send_message(request=request)

        if not isinstance(response.root, SendMessageSuccessResponse):
            # Extract error message from JSONRPC error response
            if hasattr(response.root, "error") and response.root.error:
                error_msg = getattr(
                    response.root.error, "message", str(response.root.error)
                )
                error_code = getattr(response.root.error, "code", None)
                error_str = (
                    f"Agent error (code {error_code}): {error_msg}"
                    if error_code
                    else f"Agent error: {error_msg}"
                )
                raise RuntimeError(error_str)
            raise RuntimeError("Unexpected response from agent")

        result = response.root.result
        if not isinstance(result, Message):
            raise RuntimeError("Agent returned non-message payload")
        return result

    async def stream_text(
        self,
        base_url: str,
        content: str,
        *,
        context_id: Optional[str] = None,
        task_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> AsyncGenerator[Message, None]:
        """Stream text messages to an agent.

        Args:
            base_url: Base URL of target agent
            content: Message content
            context_id: Optional context ID
            task_id: Optional task ID
            timeout: Optional timeout for streaming

        Yields:
            Messages from agent response stream
        """
        client = await self._client(base_url)
        message = Message(
            role=Role.user,
            parts=[Part(TextPart(text=content))],
            message_id=uuid.uuid4().hex,
            task_id=task_id,
            context_id=context_id,
        )
        params = MessageSendParams(message=message)
        request = SendStreamingMessageRequest(id=uuid.uuid4().hex, params=params)

        http_kwargs = {"timeout": timeout} if timeout is not None else {"timeout": None}

        async for response in client.send_message_streaming(
            request=request, http_kwargs=http_kwargs
        ):
            root = response.root
            if not isinstance(root, SendStreamingMessageSuccessResponse):
                raise RuntimeError("Unexpected streaming response from agent")
            payload = root.result
            if isinstance(payload, Message):
                yield payload

    async def wait_agent_ready(self, base_url: str, timeout: int = 10) -> bool:
        """Wait for agent to become ready by polling agent card.

        Args:
            base_url: Base URL of agent
            timeout: Maximum seconds to wait

        Returns:
            True if agent becomes ready, False on timeout
        """
        for attempt in range(1, timeout + 1):
            try:
                card = await self._resolve_card(base_url)
                if card is not None:
                    return True
            except Exception:
                await asyncio.sleep(1)
                continue
        return False

    async def aclose(self) -> None:
        """Close HTTP client."""
        await self._httpx_client.aclose()
