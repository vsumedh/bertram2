"""Minimal white agent for TextWorld evaluation.

Relies on conversation history as state - no elaborate scaffolding.
"""

from __future__ import annotations

import logging
import os
import re
from collections import OrderedDict

try:
    import tomllib as tomli  # Python 3.11+
except ModuleNotFoundError:
    import tomli  # Backport for <=3.10
from pathlib import Path
from typing import Any, Dict, List, Optional

import dotenv
import httpx
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message

from ..utils.vllm_client import VLLMClient, VLLMConfig
from ..utils.messaging import parse_tags

# Maximum number of conversation contexts to keep in memory (LRU eviction)
MAX_CONTEXTS = 100
# Number of retries for transient LLM errors
MAX_RETRIES = 2


LOGGER = logging.getLogger(__name__)
dotenv.load_dotenv()

ASSETS_DIR = Path(__file__).parent
DEFAULT_CARD_PATH = ASSETS_DIR / "agent_card.toml"

# System prompt emphasizing goal pursuit and valid action constraint
SYSTEM_PROMPT = """You are an expert TextWorld agent completing household tasks.

CRITICAL RULES:
1. Your goal is provided in the <goal> tag - work systematically to achieve it
2. You MUST select your action ONLY from the <valid_actions> list provided each turn
3. Copy the action EXACTLY as shown in valid_actions - any deviation will fail
4. You can only carry ONE object at a time
5. If an action shows "Nothing happens", the action failed - try a different approach
6. You can ONLY place/move objects at your CURRENT location - go there first!

GOAL SEMANTICS:
- Goals like "put X on cabinet" mean ANY cabinet works (cabinet 1, cabinet 2, etc.)
- Goals like "put X on shelf" mean ANY shelf works
- The goal specifies the TYPE of receptacle, not a specific instance
- Similarly, "put some saltshaker" means ANY saltshaker instance (saltshaker 1, etc.)

PLACEMENT RULE (CRITICAL):
You can ONLY place objects where you currently are. To place an object on countertop:
1. First: "go to countertop 1"
2. Then: "move [object] to countertop 1"
If you try to place an object at a location you're not at, it will fail or place it where you are!

STRATEGY FOR DIFFERENT TASK TYPES:

Pick-and-place tasks (e.g., "put X on Y"):
1. Find and pick up the target object
2. Go to the destination receptacle
3. Place the object there

Pick TWO and place tasks (e.g., "find two X and put them in Y"):
CRITICAL: You can only carry ONE object at a time! You must make TWO trips:
1. Find and pick up the FIRST object
2. Go to destination and place it
3. Go BACK to where the second object is
4. Pick up the SECOND object
5. Go to destination and place it
Do NOT try to pick up the second object while holding the first - it won't work!

Clean tasks (e.g., "put a clean X on Y"):
1. Find and pick up the object
2. Go to sinkbasin
3. Clean the object with sinkbasin
4. Go to the destination receptacle (e.g., countertop)
5. Place the clean object there

Heat tasks (e.g., "put a hot X on Y"):
1. Find and pick up the object
2. Go to microwave, open it, put object in, close, heat, open, take out
3. Go to destination and place

Cool tasks (e.g., "put a cool X on Y"):
1. Find and pick up the object
2. Go to fridge, open it, put object in, close, cool, open, take out
3. Go to destination and place

Examine/Look under lamp tasks (e.g., "look at X under the desklamp", "examine X under lamp"):
IMPORTANT: You must be holding the object AND be at the lamp location to complete this task!
1. FIRST find and pick up the target object (e.g., cd, book, mug)
2. THEN find where the desklamp is located (check desks, sidetables)
3. Go to the location WITH the desklamp (e.g., "go to desk 1" if desklamp is there)
4. Use the desklamp: "use desklamp 1"
The task completes when you use the desklamp while holding the object at that location.
Do NOT turn on the desklamp before picking up the object - find the object first!

SEARCH PRIORITIES BY OBJECT TYPE:
When searching for objects, check these locations IN ORDER:
- Cookware (pan, pot, kettle): stoveburner > countertop > cabinet > shelf
- Utensils (spatula, spoon, fork, knife, ladle): drawer > countertop > shelf
- Tableware (plate, bowl, cup, mug): cabinet > shelf > countertop > dining table
- Seasonings (saltshaker, peppershaker): shelf > countertop > drawer > cabinet
- Valuables (watch, creditcard, keychain, statue, vase, laptop, cellphone, remote):
  dresser > sidetable > shelf > desk > drawer > armchair > sofa
- Cleaning items (soapbottle, dishsponge, spraybottle, cloth):
  sinkbasin > countertop > cabinet > shelf
- Food (apple, tomato, lettuce, bread, potato, egg):
  fridge > countertop > dining table
- Books/papers (book, newspaper, pen, pencil):
  desk > shelf > sidetable > drawer > bed > sofa
- Toiletries (toiletpaper, towel, soapbar):
  toilet > cabinet > countertop
- Pillows and cushions (pillow): armchair > sofa > bed > dresser > ottoman
- Bedroom items (cd, box, basketball, baseballbat, alarmclock):
  bed > dresser > desk > shelf > drawer > sidetable

CRITICAL SEARCH RULES:
1. ALWAYS check ALL instances of each location type (shelf 1, shelf 2, shelf 3, etc.)
2. NEVER skip a location type - especially shelves, dressers, and stoveburners
3. In your reasoning, track which locations you have already visited
4. If you've checked all instances of one type, move to the next type in priority order
5. Do NOT revisit locations you have already checked unless you have new information

AVOIDING LOOPS:
If you find yourself repeating the same action multiple times:
- STOP and reconsider - you may be at the wrong location
- Check valid_actions carefully for "go to" commands
- Remember: to place an object somewhere, you must GO there first

RESPONSE FORMAT (CRITICAL - follow exactly):
You MUST respond with EXACTLY two XML tags in this order:
1. <reasoning>Your step-by-step thinking here</reasoning>
2. <command>exact action from valid_actions</command>

IMPORTANT: 
- The <reasoning> tag MUST be closed with </reasoning> before <command>
- Do NOT put any command text inside <reasoning> tags
- Do NOT include extra text outside these tags
- VERIFY your command matches your reasoning - don't say "countertop" but write "sinkbasin"
- The <command> tag must contain ONLY a single action like "go to sofa 1" - NEVER include instructions or prompt text
- If the action you want is not in valid_actions, choose the closest available action (like "go to" somewhere else)

Example response:
<reasoning>I need to find a pan. Based on search priorities for cookware, I should check stoveburners first. I'll start with stoveburner 1.</reasoning>
<command>go to stoveburner 1</command>

Example for pick-two task after picking up first object:
<reasoning>I picked up pillow 1. Since I can only carry one object, I must place it on the sofa before getting pillow 2. I'll go to the sofa now.</reasoning>
<command>go to sofa 1</command>"""


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


def _estimate_tokens(messages: List[Dict[str, str]]) -> int:
    """Rough token estimate: ~3 chars per token (conservative)."""
    return sum(len(m.get("content", "")) for m in messages) // 3


def _prune_context(
    messages: List[Dict[str, str]], max_turns: int = 8
) -> List[Dict[str, str]]:
    """Keep system message + last N turn pairs."""
    if not messages:
        return []

    result = []
    other = []

    for msg in messages:
        if msg.get("role") == "system":
            result.append(dict(msg))
        else:
            other.append(msg)

    # Keep last max_turns * 2 messages (user + assistant pairs)
    max_msgs = max_turns * 2
    if len(other) > max_msgs:
        other = other[-max_msgs:]

    result.extend(dict(m) for m in other)
    return result


class TextWorldWhiteAgentExecutor(AgentExecutor):
    """Minimal white agent - relies on conversation history as state."""

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
        temperature: float = 0.0,
        vllm_base_url: Optional[str] = None,
        max_context_turns: int = 16,
    ):
        # Bounded LRU cache for conversation contexts
        self._ctx_cache: OrderedDict[str, List[Dict[str, str]]] = OrderedDict()
        self._model = model
        self._temperature = float(temperature)
        self._max_context_turns = max_context_turns
        self._vllm_client = VLLMClient(
            VLLMConfig(
                base_url=vllm_base_url
                or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
                model=model,
            )
        )

    def _get_messages(self, context_id: str) -> List[Dict[str, str]]:
        """Get or create conversation history with LRU eviction."""
        if context_id in self._ctx_cache:
            # Move to end (most recently used)
            self._ctx_cache.move_to_end(context_id)
            return self._ctx_cache[context_id]

        # Evict oldest if at capacity
        if len(self._ctx_cache) >= MAX_CONTEXTS:
            self._ctx_cache.popitem(last=False)

        self._ctx_cache[context_id] = []
        return self._ctx_cache[context_id]

    def _select_valid_action(self, intended: str, valid_actions_text: str) -> str:
        """Select best matching action from valid_actions list.

        Priority: exact match > same-verb match > substring match > fallback

        Args:
            intended: The action the LLM intended to take
            valid_actions_text: Raw text of valid actions (newline-separated)

        Returns:
            A valid action string guaranteed to be in valid_actions
        """
        # Parse valid actions (handle "- " and "* " prefixes)
        valid_actions = []
        if valid_actions_text.strip():
            for line in valid_actions_text.strip().split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    line = line[2:]
                elif line.startswith("* "):
                    line = line[2:]
                if line and line[0].isalpha():
                    valid_actions.append(line.strip())

        # If no valid actions parsed, return intended as-is
        if not valid_actions:
            return intended

        # Detect clearly invalid commands (prompt text, instructions, etc.)
        # These should never be executed - force fallback
        intended_lower = intended.lower().strip()
        invalid_patterns = [
            "must choose",
            "you must",
            "from this list",
            "valid_actions",
            "choose exactly",
            "select one",
        ]
        is_clearly_invalid = (
            len(intended) > 50  # Commands are typically short
            or any(pattern in intended_lower for pattern in invalid_patterns)
            or not intended.strip()
        )
        if is_clearly_invalid:
            LOGGER.warning(
                f"[White] Detected invalid command (prompt text?): '{intended[:50]}...', "
                f"falling back to first valid action"
            )
            return valid_actions[0]

        # Exact match (case-insensitive)
        for va in valid_actions:
            if va.lower().strip() == intended_lower:
                return va

        # Same-verb preference (e.g., "go to X" -> "go to Y")
        intended_parts = intended_lower.split()
        if intended_parts:
            verb = intended_parts[0]
            for va in valid_actions:
                if va.lower().startswith(verb + " "):
                    return va

        # Substring match
        for va in valid_actions:
            va_lower = va.lower()
            if intended_lower in va_lower or va_lower in intended_lower:
                return va

        # Fallback to first valid action
        return valid_actions[0]

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Process observation and generate action."""
        user_input = context.get_user_input()

        # Debug: log the context_id to trace state management
        LOGGER.debug(
            f"[White] Received request with context_id={context.context_id}, "
            f"cache keys={list(self._ctx_cache.keys())}"
        )

        # Parse structured tags from green agent message
        tags = parse_tags(user_input)
        goal = tags.get("goal", "")
        observation = tags.get("observation", user_input)
        valid_actions = tags.get("valid_actions", "")

        # Get conversation history early (needed for failed action check)
        messages = self._get_messages(context.context_id)

        # Build clean user content from parsed tags, emphasizing valid actions
        parts = []
        if goal:
            parts.append(f"GOAL: {goal}")

        # Annotate failed actions to prevent repetition
        if "nothing happens" in observation.lower():
            # Find last action from conversation history
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    last_tags = parse_tags(msg.get("content", ""))
                    last_cmd = last_tags.get("command", "")
                    if last_cmd:
                        parts.append(
                            f"WARNING: Your previous action '{last_cmd}' FAILED with 'Nothing happens'. "
                            f"Do NOT repeat this action. Choose a DIFFERENT action from the list."
                        )
                        LOGGER.debug(f"[White] Annotated failed action: {last_cmd}")
                    break

        parts.append(f"OBSERVATION: {observation}")
        if valid_actions:
            parts.append(
                f"VALID ACTIONS (you MUST choose from this list):\n{valid_actions}"
            )
        user_content = "\n\n".join(parts)

        LOGGER.debug(
            f"[White] context_id={context.context_id}, history length={len(messages)}"
        )

        # Add system prompt on first turn
        if not messages:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})

        # Add user message with structured content
        messages.append({"role": "user", "content": user_content})

        # Prune if needed and update cache directly (avoid in-place mutation)
        pruned = _prune_context(messages, self._max_context_turns)
        self._ctx_cache[context.context_id] = pruned
        messages = pruned

        # Calculate safe max_tokens
        input_tokens = _estimate_tokens(messages)
        max_tokens = min(512, max(256, 8192 - input_tokens - 100))

        # Generate response with retry logic and action validation
        response_text = await self._generate_with_retry(
            messages, max_tokens, valid_actions
        )

        messages.append({"role": "assistant", "content": response_text})

        await event_queue.enqueue_event(
            new_agent_text_message(response_text, context_id=context.context_id)
        )

    async def _generate_with_retry(
        self, messages: List[Dict[str, str]], max_tokens: int, valid_actions: str = ""
    ) -> str:
        """Generate LLM response with retry logic for transient errors.

        Args:
            messages: Conversation history
            max_tokens: Maximum tokens to generate
            valid_actions: Raw text of valid actions for validation

        Returns:
            Formatted response with guaranteed valid action
        """
        last_error: Optional[Exception] = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await self._vllm_client.acompletion(
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=max_tokens,
                    stop=["</command>"],  # Stop at command end to prevent runaway
                )
                content = response.content
                # Restore closing tag consumed by stop sequence
                if "<command>" in content and "</command>" not in content:
                    content += "</command>"

                formatted = self._ensure_format(content)

                # Always validate and correct action against valid_actions list
                resp_tags = parse_tags(formatted)
                cmd = resp_tags.get("command", "")

                # Detect clearly invalid commands (prompt text, instructions, etc.)
                # These should never be executed - force fallback even without valid_actions
                cmd_lower = cmd.lower().strip()
                invalid_patterns = [
                    "must choose",
                    "you must",
                    "from this list",
                    "valid_actions",
                    "choose exactly",
                    "select one",
                    "action from",
                    "choose from",
                ]
                is_clearly_invalid = (
                    len(cmd) > 50  # Commands are typically short
                    or any(pattern in cmd_lower for pattern in invalid_patterns)
                    or not cmd.strip()
                )

                if is_clearly_invalid:
                    LOGGER.warning(
                        f"[White] Detected invalid command: '{cmd[:50]}...', forcing fallback"
                    )
                    # Use first valid action if available, otherwise "look"
                    if valid_actions:
                        corrected_cmd = self._select_valid_action("look", valid_actions)
                    else:
                        corrected_cmd = "look"
                    reasoning = resp_tags.get("reasoning", "")
                    formatted = f"<reasoning>{reasoning}</reasoning><command>{corrected_cmd}</command>"
                elif valid_actions:
                    corrected_cmd = self._select_valid_action(cmd, valid_actions)
                    if corrected_cmd != cmd:
                        LOGGER.info(
                            f"[White] Corrected action '{cmd}' -> '{corrected_cmd}'"
                        )
                        reasoning = resp_tags.get("reasoning", "")
                        formatted = f"<reasoning>{reasoning}</reasoning><command>{corrected_cmd}</command>"

                return formatted
            except httpx.TimeoutException as exc:
                LOGGER.warning(
                    f"LLM timeout (attempt {attempt + 1}/{MAX_RETRIES + 1}): {exc}"
                )
                last_error = exc
                # Continue to retry
            except httpx.HTTPStatusError as exc:
                LOGGER.error(f"LLM HTTP error: {exc.response.status_code}")
                last_error = exc
                break  # Don't retry HTTP errors
            except Exception as exc:
                LOGGER.error(f"LLM call failed: {type(exc).__name__}: {exc}")
                last_error = exc
                break  # Don't retry unknown errors

        # All retries exhausted or non-retryable error
        LOGGER.error(
            f"LLM generation failed after {attempt + 1} attempt(s): {last_error}"
        )
        return "<reasoning>Error during generation</reasoning><command>look</command>"

    def _ensure_format(self, content: str) -> str:
        """Ensure response has proper tags."""
        has_reasoning = "<reasoning>" in content
        has_command = "<command>" in content

        if has_reasoning and has_command:
            return content

        if not has_command:
            # Try to extract a command from raw text
            cmd = self._extract_command_from_raw(content)
            if has_reasoning:
                return f"{content}<command>{cmd}</command>"
            return f"<reasoning>{content.strip()}</reasoning><command>{cmd}</command>"

        # has command but no reasoning
        return f"<reasoning></reasoning>{content}"

    def _extract_command_from_raw(self, text: str) -> str:
        """Extract command from unformatted text."""
        text_lower = text.lower()

        # Common ALFWorld action patterns
        patterns = [
            (r"go to ([\w\s]+\d+)", "go to {}"),
            (r"take ([\w\s]+\d+) from ([\w\s]+\d+)", "take {} from {}"),
            (r"take ([\w\s]+\d+)", "take {}"),
            (r"put ([\w\s]+\d+) (?:in|on) ([\w\s]+\d+)", "put {} in/on {}"),
            (r"move ([\w\s]+\d+) to ([\w\s]+\d+)", "move {} to {}"),
            (r"open ([\w\s]+\d+)", "open {}"),
            (r"close ([\w\s]+\d+)", "close {}"),
            (r"examine ([\w\s]+\d+)", "examine {}"),
            (r"use ([\w\s]+\d+)", "use {}"),
            (r"clean ([\w\s]+\d+) with ([\w\s]+\d+)", "clean {} with {}"),
            (r"heat ([\w\s]+\d+) with ([\w\s]+\d+)", "heat {} with {}"),
            (r"cool ([\w\s]+\d+) with ([\w\s]+\d+)", "cool {} with {}"),
            (r"slice ([\w\s]+\d+) with ([\w\s]+\d+)", "slice {} with {}"),
        ]

        for pattern, template in patterns:
            match = re.search(pattern, text_lower)
            if match:
                groups = match.groups()
                if len(groups) == 1:
                    return template.format(groups[0].strip())
                elif len(groups) == 2:
                    return template.format(groups[0].strip(), groups[1].strip())

        for cmd in ["look", "inventory"]:
            if cmd in text_lower:
                return cmd

        return "look"

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
    vllm_base_url: str | None = None,
    **_kwargs,  # Ignore legacy kwargs
) -> None:
    """Start the white agent HTTP service."""
    demo_mode = os.environ.get("DEMO_MODE", "0") == "1"
    # Use DEBUG level when GREEN_VERBOSE is set (propagates from launcher -v flag)
    verbose_mode = os.environ.get("GREEN_VERBOSE") == "1"
    level = (
        logging.DEBUG
        if verbose_mode
        else (logging.WARNING if demo_mode else logging.INFO)
    )
    logging.basicConfig(level=level)
    if demo_mode:
        try:
            logging.disable(logging.CRITICAL)
        except Exception:
            pass

    for noisy in (
        "httpx",
        "httpcore",
        "httpcore.connection",
        "httpcore.http11",
        "a2a",
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
    ):
        try:
            logging.getLogger(noisy).setLevel(logging.WARNING)
        except Exception:
            pass

    LOGGER.info("Starting TextWorld white agent on %s:%s", host, port)

    # Load agent card
    card_path = ASSETS_DIR / f"{agent_name}.toml"
    if card_path.exists():
        card_dict = load_agent_card_toml(card_path)
        card_dict["url"] = f"http://{host}:{port}"
        card = AgentCard(**card_dict)
    else:
        url = f"http://{host}:{port}"
        card = prepare_white_agent_card(url)

    # Resolve config from args or env
    resolved_model = model or os.environ.get(
        "WHITE_MODEL", "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"
    )
    resolved_temp = float(
        temperature
        if temperature is not None
        else os.environ.get("WHITE_TEMPERATURE", 0.0)
    )
    resolved_vllm_url = vllm_base_url or os.environ.get(
        "VLLM_BASE_URL", "http://localhost:8000/v1"
    )

    executor = TextWorldWhiteAgentExecutor(
        model=resolved_model,
        temperature=resolved_temp,
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

    uvicorn.run(
        application.build(), host=host, port=port, log_level="warning", access_log=False
    )


__all__ = ["TextWorldWhiteAgentExecutor", "start_white_agent"]
