"""Hardcoded white agent implementation using observation-aware strategies.

This agent uses observation-aware strategy execution for ALFWorld tasks,
adapting to the environment state rather than blindly following pre-recorded
trajectories. It parses observations to detect objects, tracks state, and
selects actions from the valid_actions list provided by the environment.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict

try:
    import tomllib as tomli  # Python 3.11+
except ModuleNotFoundError:
    import tomli  # Backport for <=3.10
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Maximum number of conversation contexts to keep in memory (LRU eviction)
MAX_CONTEXTS = 100

import dotenv
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message

from ..utils.messaging import parse_tags
from .hardcoded_trajectories import (
    identify_task_type,
    generate_reasoning,
    get_profile,
    AGENT_PROFILES,
    EXPERT_TRAJECTORIES,
    match_task,
    find_action_in_valid,
    # New observation-aware functions
    AgentState,
    initialize_state_from_goal,
    select_adaptive_action,
    parse_valid_actions,
    parse_observation,
)


LOGGER = logging.getLogger(__name__)
dotenv.load_dotenv()

ASSETS_DIR = Path(__file__).parent
DEFAULT_CARD_PATH = ASSETS_DIR / "agent_card_hardcoded.toml"


def load_agent_card_toml(card_path: Path = DEFAULT_CARD_PATH) -> Dict[str, Any]:
    """Load agent card from TOML file."""
    with card_path.open("rb") as fh:
        return tomli.load(fh)


def prepare_hardcoded_agent_card(url: str) -> AgentCard:
    """Prepare agent card for hardcoded agent."""
    skill = AgentSkill(
        id="textworld_hardcoded_policy",
        name="TextWorld Hardcoded Policy",
        description="Completes household tasks using pre-recorded expert trajectories",
        tags=["textworld", "household", "hardcoded", "optimal"],
        examples=[],
    )
    card = AgentCard(
        name="textworld_hardcoded_agent",
        description="Hardcoded white agent using pre-recorded optimal trajectories",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


class HardcodedWhiteAgentExecutor(AgentExecutor):
    """Observation-aware white agent with deterministic profile-driven behavior.

    This agent adapts to the environment by:
    1. Parsing observations to detect objects and locations
    2. Tracking state (inventory, visited locations, task progress)
    3. Selecting actions from the valid_actions list
    4. Following task-type-specific strategies
    """

    def __init__(self, profile: str = "expert"):
        """Initialize hardcoded agent executor with a profile.

        Args:
            profile: Agent profile name (expert, competent, novice, lucky_guesser, overthinker)
        """
        self.profile_name = profile
        self.profile = get_profile(profile)

        # Bounded LRU cache for conversation contexts (now stores AgentState)
        self._ctx_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

        LOGGER.info(
            "Initialized observation-aware hardcoded agent with profile '%s': "
            "noise=%s detour=%s look_after_goto=%s reasoning=%s",
            profile,
            self.profile.get("search_priority_noise_mode"),
            self.profile.get("detour_every_n_steps"),
            self.profile.get("extra_look_after_goto"),
            self.profile.get("reasoning_style"),
        )

    def _get_state(self, context_id: str, goal: str) -> Dict[str, Any]:
        """Get or create conversation state with LRU eviction."""
        if context_id in self._ctx_cache:
            # Move to end (most recently used)
            self._ctx_cache.move_to_end(context_id)
            return self._ctx_cache[context_id]

        # Evict oldest if at capacity
        if len(self._ctx_cache) >= MAX_CONTEXTS:
            self._ctx_cache.popitem(last=False)

        # Create new state with AgentState for adaptive execution
        agent_state = initialize_state_from_goal(goal) if goal else AgentState()
        self._ctx_cache[context_id] = {
            "step": 0,
            "goal": goal,
            "agent_state": agent_state,
            "last_action": None,
            "consecutive_looks": 0,
            "matched_task_id": -1,  # For trajectory preference
        }
        return self._ctx_cache[context_id]

    def _select_with_trajectory_preference(
        self,
        agent_state: AgentState,
        observation: str,
        valid_actions: List[str],
        step: int,
        goal: str,
        matched_task_id: int,
    ) -> Tuple[str, str, int]:
        """Select action preferring pre-recorded trajectory when valid.

        Falls back to adaptive selection when trajectory action is invalid.

        Returns:
            Tuple of (action, phase_hint, updated_task_id)
        """
        # Try to match current task to a pre-recorded trajectory on first step
        task_id = matched_task_id
        if task_id < 0 and step == 0:
            task_id = match_task(goal, observation)
            LOGGER.debug(f"Matched task_id={task_id} for goal: {goal[:50]}...")

        if task_id >= 0 and task_id in EXPERT_TRAJECTORIES:
            trajectory = EXPERT_TRAJECTORIES[task_id]["actions"]
            if step < len(trajectory):
                intended_action = trajectory[step]
                # Check if this action is valid in current environment
                matched = find_action_in_valid(intended_action, valid_actions)
                if matched:
                    LOGGER.debug(
                        f"Using pre-recorded action at step {step}: {matched}"
                    )
                    return matched, "expert_trajectory", task_id
                else:
                    LOGGER.debug(
                        f"Pre-recorded action '{intended_action}' not valid, "
                        f"falling back to adaptive"
                    )

        # Fall back to adaptive selection
        action, phase_hint = select_adaptive_action(
            agent_state,
            observation,
            valid_actions,
            step=step,
            profile=self.profile,
        )
        return action, phase_hint, task_id

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Process observation and return adaptive action with reasoning."""
        user_input = context.get_user_input()
        tags = parse_tags(user_input)

        # Extract goal, observation, and valid_actions
        goal = tags.get("goal", "")
        observation = tags.get("observation", "")
        valid_actions_text = tags.get("valid_actions", "")

        # Parse valid actions from the tag
        valid_actions = parse_valid_actions(valid_actions_text)

        # Get or initialize conversation state
        state = self._get_state(context.context_id, goal)

        # Update goal if this is a new/different goal
        if goal and goal != state["goal"]:
            state["goal"] = goal
            state["agent_state"] = initialize_state_from_goal(goal)
            state["step"] = 0
            state["consecutive_looks"] = 0
            state["matched_task_id"] = -1  # Reset task matching
            LOGGER.info(
                f"Initialized state for goal: {goal[:50]}... "
                f"(type={state['agent_state'].task_type}, "
                f"object={state['agent_state'].target_object}, "
                f"target={state['agent_state'].target_receptacle})"
            )

        agent_state: AgentState = state["agent_state"]
        current_step = state["step"]
        stored_goal = state["goal"] or goal
        matched_task_id = state.get("matched_task_id", -1)
        # Keep AgentState.last_action synchronized for deterministic profile behaviors.
        agent_state.last_action = state.get("last_action")

        # Select action: prefer pre-recorded trajectory if enabled and valid
        if self.profile.get("prefer_prerecorded_trajectory"):
            action, phase_hint, matched_task_id = self._select_with_trajectory_preference(
                agent_state=agent_state,
                observation=observation,
                valid_actions=valid_actions,
                step=current_step,
                goal=stored_goal,
                matched_task_id=matched_task_id,
            )
            state["matched_task_id"] = matched_task_id
        else:
            # Select action using observation-aware strategy
            action, phase_hint = select_adaptive_action(
                agent_state,
                observation,
                valid_actions,
                step=current_step,
                profile=self.profile,
            )

        # Filter valid_actions to only include actual game commands
        # (exclude prompt text that might have leaked into the list)
        game_commands = [
            a
            for a in valid_actions
            if a.lower().startswith(
                (
                    "go to ",
                    "take ",
                    "move ",
                    "put ",
                    "open ",
                    "close ",
                    "use ",
                    "examine ",
                    "clean ",
                    "heat ",
                    "cool ",
                )
            )
            or a.lower() in ("look", "inventory")
        ]

        # Prevent infinite look loops
        if action.lower() == "look":
            state["consecutive_looks"] += 1
            if state["consecutive_looks"] > 2:
                # Force a different action if stuck looking
                for alt_action in game_commands:
                    if (
                        alt_action.lower() != "look"
                        and alt_action.lower() != "inventory"
                    ):
                        action = alt_action
                        state["consecutive_looks"] = 0
                        LOGGER.info(f"Breaking look loop, switching to: {action}")
                        break
        else:
            state["consecutive_looks"] = 0

        # Validate action is in valid_actions (should always be true with new logic)
        if game_commands and action not in game_commands:
            # Try case-insensitive match
            action_lower = action.lower()
            matched = False
            for va in game_commands:
                if va.lower() == action_lower:
                    action = va
                    matched = True
                    break
            if not matched:
                # Fallback to first valid game command
                LOGGER.warning(
                    f"Action '{action}' not in valid_actions, "
                    f"falling back to '{game_commands[0]}'"
                )
                action = game_commands[0]

        # Generate reasoning based on deterministic profile policy
        reasoning = generate_reasoning(
            action,
            current_step,
            stored_goal,
            observation,
            self.profile,
        )

        # Update state
        state["step"] += 1
        state["last_action"] = action
        agent_state.last_action = action

        # Format response with reasoning and command tags
        response_text = f"<reasoning>{reasoning}</reasoning><command>{action}</command>"

        LOGGER.info(
            f"Step {current_step}: phase={phase_hint}, action='{action}', "
            f"acquired={agent_state.object_acquired}, placed={agent_state.object_placed}"
        )

        await event_queue.enqueue_event(
            new_agent_text_message(response_text, context_id=context.context_id)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle cancellation request."""
        await event_queue.enqueue_event(
            new_agent_text_message("Cancellation not implemented.")
        )


def start_hardcoded_agent(
    *,
    agent_name: str = "agent_card_hardcoded",
    host: str = "0.0.0.0",
    port: int = 8725,
    profile: str = "expert",
) -> None:
    """Start the hardcoded agent HTTP service.

    Args:
        agent_name: Name of the agent card TOML file
        host: Host to bind to
        port: Port to listen on
        profile: Agent profile (expert, competent, novice, lucky_guesser, overthinker)
    """
    demo_mode = os.environ.get("DEMO_MODE", "0") == "1"

    # Configure logging
    logging.basicConfig(level=(logging.WARNING if demo_mode else logging.INFO))
    if demo_mode:
        try:
            logging.disable(logging.CRITICAL)
        except Exception:
            pass

    # Suppress noisy third-party logs
    for noisy in ("httpx", "a2a", "uvicorn", "uvicorn.error", "uvicorn.access"):
        try:
            logging.getLogger(noisy).setLevel(logging.WARNING)
        except Exception:
            pass

    LOGGER.info(
        "Starting TextWorld hardcoded agent on %s:%s (profile=%s)", host, port, profile
    )

    # Load agent card
    card_path = ASSETS_DIR / f"{agent_name}.toml"
    if card_path.exists():
        card_dict = load_agent_card_toml(card_path)
        card_dict["url"] = f"http://{host}:{port}"
        card = AgentCard(**card_dict)
    else:
        url = f"http://{host}:{port}"
        card = prepare_hardcoded_agent_card(url)

    executor = HardcodedWhiteAgentExecutor(profile=profile)
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


__all__ = ["HardcodedWhiteAgentExecutor", "start_hardcoded_agent"]
