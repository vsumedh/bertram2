"""Hardcoded white agent implementation using pre-recorded trajectories.

This agent uses pre-recorded expert action sequences for ALFWorld tasks,
identifying the appropriate trajectory based on goal and observation patterns.
"""

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

from ..utils.messaging import parse_tags
from .hardcoded_trajectories import (
    match_task,
    get_action_with_reasoning,
    identify_task_type,
    generate_reasoning,
    degrade_trajectory,
    get_profile,
    AGENT_PROFILES,
    EXPERT_TRAJECTORIES,
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
    """Hardcoded white agent that replays pre-recorded trajectories with configurable quality."""

    def __init__(self, profile: str = "expert"):
        """Initialize hardcoded agent executor with a profile.

        Args:
            profile: Agent profile name (expert, competent, novice, lucky_guesser, overthinker)
        """
        self.profile_name = profile
        self.profile = get_profile(profile)
        self.reasoning_quality = self.profile["reasoning"]
        self.strategy_quality = self.profile["strategy"]

        # Pre-compute degraded trajectories for this profile
        self.degraded_trajectories: Dict[int, list] = {}
        for task_id, task_data in EXPERT_TRAJECTORIES.items():
            self.degraded_trajectories[task_id] = degrade_trajectory(
                task_data["actions"],
                self.strategy_quality,
                seed=task_id,  # Use task_id as seed for reproducibility
            )

        # Track state per conversation context
        self.ctx_id_to_state: Dict[str, Dict[str, Any]] = {}

        LOGGER.info(
            f"Initialized hardcoded agent with profile '{profile}': "
            f"reasoning={self.reasoning_quality}, strategy={self.strategy_quality}"
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Process observation and return expert action with reasoning."""
        user_input = context.get_user_input()
        tags = parse_tags(user_input)

        # Extract goal and observation
        goal = tags.get("goal", "")
        observation = tags.get("observation", "")

        # Get or initialize conversation state
        if context.context_id not in self.ctx_id_to_state:
            self.ctx_id_to_state[context.context_id] = {
                "step": 0,
                "task_id": None,
                "goal": goal,
                "initial_observation": observation,
            }

        state = self.ctx_id_to_state[context.context_id]

        # On first message with goal, identify the task
        if state["task_id"] is None and goal:
            state["goal"] = goal
            state["initial_observation"] = observation
            state["task_id"] = match_task(goal, observation)
            LOGGER.info(
                f"Identified task_id={state['task_id']} for goal: {goal[:50]}..."
            )

        # Get action and reasoning for current step
        current_step = state["step"]
        task_id = state["task_id"]
        stored_goal = state["goal"] or goal

        if task_id is not None and task_id >= 0:
            # Use pre-degraded trajectory for this task
            trajectory = self.degraded_trajectories.get(task_id)
            action, reasoning = get_action_with_reasoning(
                task_id,
                current_step,
                stored_goal,
                observation,
                reasoning_quality=self.reasoning_quality,
                trajectory_override=trajectory,
            )
        else:
            # Fallback: use generic exploration
            action = "look"
            reasoning = self._generate_fallback_reasoning(
                current_step, stored_goal, observation
            )

        # Increment step counter
        state["step"] += 1

        # Format response with reasoning and command tags
        response_text = f"<reasoning>{reasoning}</reasoning><command>{action}</command>"

        LOGGER.info(
            f"Step {current_step}: action='{action}', reasoning='{reasoning[:50]}...'"
        )

        await event_queue.enqueue_event(
            new_agent_text_message(response_text, context_id=context.context_id)
        )

    def _generate_fallback_reasoning(
        self, step: int, goal: str, observation: str
    ) -> str:
        """Generate fallback reasoning when task matching fails."""
        # Use reasoning quality to determine fallback style
        if self.reasoning_quality == "low":
            return generate_reasoning("look", step, goal, observation, "low")
        elif self.reasoning_quality == "medium":
            return generate_reasoning("look", step, goal, observation, "medium")

        # High quality fallback
        task_type = identify_task_type(goal)
        fallback_templates = [
            f"Exploring the environment to locate objects needed for the {task_type} task.",
            f"Surveying available locations to plan the approach for completing the goal.",
            f"Gathering information about the current state to determine the next step.",
            f"Observing surroundings to identify relevant items for the task objective.",
            f"Examining the area systematically to progress toward goal completion.",
        ]

        return fallback_templates[step % len(fallback_templates)]

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
