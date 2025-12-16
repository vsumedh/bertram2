"""White agent implementation for TextWorld evaluation."""

from __future__ import annotations

import logging
import os
try:
    import tomllib as tomli  # Python 3.11+
except ModuleNotFoundError:
    import tomli  # Backport for <=3.10
from pathlib import Path
from typing import Any, Dict, Optional

import dotenv
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message
from ..utils.vllm_client import VLLMClient, VLLMConfig
from ..utils.messaging import parse_tags


LOGGER = logging.getLogger(__name__)
dotenv.load_dotenv()

ASSETS_DIR = Path(__file__).parent
DEFAULT_CARD_PATH = ASSETS_DIR / "agent_card.toml"


def load_agent_card_toml(card_path: Path = DEFAULT_CARD_PATH) -> Dict[str, Any]:
    """Load agent card from TOML file."""
    with card_path.open("rb") as fh:
        return tomli.load(fh)


def prepare_white_agent_card(url: str) -> AgentCard:
    """Prepare agent card for white agent."""
    skill = AgentSkill(
        id="textworld_task_fulfillment",
        name="TextWorld Task Fulfillment",
        description="Completes household tasks in TextWorld environment",
        tags=["textworld", "household", "general"],
        examples=[],
    )
    card = AgentCard(
        name="textworld_white_agent",
        description="White agent for TextWorld evaluation",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


class TextWorldWhiteAgentExecutor(AgentExecutor):
    """White agent that plays TextWorld games using LLM policy."""

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.0,
        prompt_profile: str = "standard",
        vllm_base_url: Optional[str] = None,
    ):
        self.ctx_id_to_messages = {}
        self._model = model
        self._temperature = float(temperature)
        self._prompt_profile = prompt_profile
        self._vllm_client = VLLMClient(VLLMConfig(
            base_url=vllm_base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
            model=model,
        ))

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Process observation and generate action."""
        user_input = context.get_user_input()

        # Maintain conversation history
        if context.context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context.context_id] = []
        messages = self.ctx_id_to_messages[context.context_id]

        # Check message type
        tags = parse_tags(user_input)

        if "observation" in tags or "goal" in tags:
            # Active gameplay - generate command
            response_text = await self._generate_command(user_input, messages)
        else:
            # Generic response
            messages.append({"role": "user", "content": user_input})
            try:
                llm_response = self._vllm_client.completion(
                    messages=messages,
                    temperature=self._temperature,
                )
                response_text = llm_response.content
                messages.append({"role": "assistant", "content": response_text})
            except Exception as exc:
                LOGGER.error(f"LLM call failed: {exc}")
                response_text = "<command>look</command>"

        await event_queue.enqueue_event(
            new_agent_text_message(response_text, context_id=context.context_id)
        )

    async def _generate_command(self, user_input: str, messages: list) -> str:
        """Generate command for TextWorld."""
        # Add system message on first turn
        if len(messages) == 0:
            if self._prompt_profile == "concise":
                sys_content = """You are a concise, expert TextWorld agent.

Respond ONLY with:
<reasoning>brief rationale</reasoning><command>single command</command>"""
            else:
                sys_content = """You are an expert TextWorld agent completing household tasks.

Common actions:
- Navigation: "go to [location]"
- Take/drop: "take [object]", "drop [object]"
- Containers: "open [container]", "close [container]"
- Placement: "put [object] in/on [location]"
- Examination: "examine [object]", "inventory", "look"
- Special: "clean [object] with [tool]", "heat [object] with [appliance]", "cool [object] with [appliance]"

Respond with your reasoning first in <reasoning>...</reasoning> tags explaining why you chose this action, then your command in <command>...</command> tags.
Format: <reasoning>Your thinking process here</reasoning><command>go to kitchen</command>"""
            system_msg = {
                "role": "system",
                "content": sys_content,
            }
            messages.append(system_msg)

        # Add user observation
        messages.append({"role": "user", "content": user_input})

        # Generate action
        try:
            response = self._vllm_client.completion(
                messages=messages,
                temperature=self._temperature,
            )

            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            # Ensure proper formatting - check for reasoning and command tags
            has_reasoning = "<reasoning>" in assistant_content
            has_command = "<command>" in assistant_content

            if not has_reasoning and not has_command:
                # Neither tag present, wrap everything as reasoning + command
                assistant_content = f"<reasoning>{assistant_content.strip()}</reasoning><command>look</command>"
            elif not has_reasoning:
                # Has command but no reasoning, add empty reasoning
                assistant_content = f"<reasoning></reasoning>{assistant_content}"
            elif not has_command:
                # Has reasoning but no command, extract text after reasoning and wrap as command
                # If we can't extract cleanly, default to look
                assistant_content = f"{assistant_content}<command>look</command>"

            return assistant_content

        except Exception as exc:
            LOGGER.error(f"LLM call failed: {exc}")
            return "<reasoning>Error occurred during command generation</reasoning><command>look</command>"

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(
            new_agent_text_message("Cancellation not implemented.")
        )


def start_white_agent(
    *,
    agent_name: str = "agent_card",
    host: str = "0.0.0.0",
    port: int = 9002,
    model: str | None = None,
    temperature: float | None = None,
    prompt_profile: str | None = None,
    vllm_base_url: str | None = None,
) -> None:
    """Start the white agent HTTP service."""
    demo_mode = os.environ.get("DEMO_MODE", "0") == "1"
    logging.basicConfig(level=(logging.WARNING if demo_mode else logging.INFO))
    if demo_mode:
        # Mute virtually all logs in demo mode; green agent will print the exchange
        try:
            logging.disable(logging.CRITICAL)
        except Exception:
            pass
    # Suppress noisy third-party logs for demo-friendly output
    for noisy in ("httpx", "a2a", "uvicorn", "uvicorn.error", "uvicorn.access"):
        try:
            logging.getLogger(noisy).setLevel(logging.WARNING)
        except Exception:
            pass
    LOGGER.info("Starting TextWorld white agent on %s:%s", host, port)

    # Try to load from TOML, otherwise generate dynamically
    card_path = ASSETS_DIR / f"{agent_name}.toml"
    if card_path.exists():
        card_dict = load_agent_card_toml(card_path)
        card_dict["url"] = f"http://{host}:{port}"
        card = AgentCard(**card_dict)
    else:
        url = f"http://{host}:{port}"
        card = prepare_white_agent_card(url)

    # Resolve overrides from args or env
    resolved_model = model or os.environ.get("WHITE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    resolved_temp = float(temperature if temperature is not None else os.environ.get("WHITE_TEMPERATURE", 0.0))
    resolved_profile = (prompt_profile or os.environ.get("WHITE_PROMPT_PROFILE", "standard")).strip().lower()
    resolved_vllm_url = vllm_base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")

    executor = TextWorldWhiteAgentExecutor(
        model=resolved_model,
        temperature=resolved_temp,
        prompt_profile=resolved_profile,
        vllm_base_url=resolved_vllm_url,
    )
    handler = DefaultRequestHandler(
        agent_executor=executor, task_store=InMemoryTaskStore()
    )
    application = A2AStarletteApplication(
        agent_card=card,
        http_handler=handler,
    )

    import uvicorn

    uvicorn.run(application.build(), host=host, port=port, log_level="warning", access_log=False)


__all__ = ["TextWorldWhiteAgentExecutor", "start_white_agent"]
