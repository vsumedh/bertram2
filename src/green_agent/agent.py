"""Green agent implementation for TextWorld evaluation."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import tomllib as tomli  # Python 3.11+
except ModuleNotFoundError:
    import tomli  # Backport for <=3.10

import dotenv
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard
from a2a.utils import new_agent_text_message

from ..utils.a2a_client import A2AMessenger
from ..utils.messaging import parse_tags
from ..utils.textworld_env import (
    ExpertTrajectory,
    TaskConfig,
    TextWorldEnvironment,
    TextWorldEnvironmentError,
)
from ..utils.ground_truth_baseline import GroundTruthTrajectory
from .green_assessor import (
    STEP_BUDGET,
    evaluate as unified_evaluate,
)
from .rubric import load_rubric
from .episode_runner import EpisodeRunner
from .output_formatter import DemoFormatter, CollectingFormatter


LOGGER = logging.getLogger(__name__)
dotenv.load_dotenv()

ASSETS_DIR = Path(__file__).parent
DEFAULT_AGENT_CARD_TOML = ASSETS_DIR / "agent_card.toml"


class GreenAgentError(Exception):
    """Domain-specific error for green agent."""


def load_agent_card_toml(
    agent_name: str, directory: Path = ASSETS_DIR
) -> Dict[str, Any]:
    """Load agent card from TOML file."""
    card_path = directory / f"{agent_name}.toml"
    if not card_path.exists():
        card_path = DEFAULT_AGENT_CARD_TOML
    with card_path.open("rb") as fh:
        return tomli.load(fh)


class TextWorldGreenAgentExecutor(AgentExecutor):
    """Green agent that manages environment and evaluates white agent."""

    def __init__(self, rubric_path: Optional[Path] = None) -> None:
        """Initialize green agent executor.

        Args:
            rubric_path: Optional path to evaluation rubric file. If not provided,
                will use default location or environment variable.
        """
        self._messenger = A2AMessenger()

        # Load evaluation rubric with fallback handling
        try:
            self._rubric = load_rubric(rubric_path)
            LOGGER.info(
                f"Loaded evaluation rubric with {len(self._rubric.criteria)} criteria"
            )
        except Exception as exc:
            LOGGER.warning(
                f"Failed to load evaluation rubric: {exc}. Using fallback evaluation."
            )
            self._rubric = None
        # Demo mode: only print human-friendly green/white exchanges
        self._demo = os.environ.get("DEMO_MODE", "0") == "1"

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Main execution loop for evaluation."""
        # Parse incoming task
        user_input = context.get_user_input()
        tags = parse_tags(user_input)

        white_agent_url = tags.get("white_agent_url")
        if not white_agent_url:
            raise GreenAgentError("Task payload missing <white_agent_url> tag")
        raw_task_config = tags.get("task_config") or tags.get("env_config") or "{}"
        benchmark_mode = tags.get("benchmark_mode", "0").strip() == "1"
        try:
            task_config = TaskConfig.from_payload(json.loads(raw_task_config))
        except json.JSONDecodeError as exc:
            raise GreenAgentError("Invalid JSON in <task_config>") from exc

        # Setup environment - request expert plan for fallback baseline
        env = TextWorldEnvironment(task_config, use_expert_plan=True)
        try:
            setup_payload = env.setup()
        except TextWorldEnvironmentError as exc:
            raise GreenAgentError(str(exc)) from exc

        goal = setup_payload["goal"]
        observation = setup_payload["initial_observation"]

        # Load ground truth trajectory from traj_data.json (optimal human demonstration)
        # This is the ideal baseline representing actual human-level optimal performance.
        LOGGER.info("Loading ground truth trajectory from traj_data.json...")
        ground_truth: Optional[GroundTruthTrajectory] = None
        try:
            ground_truth = env.get_ground_truth_trajectory()
            LOGGER.info(
                f"Ground truth trajectory loaded: {ground_truth.step_count} steps, "
                f"task_type={ground_truth.task_type}"
            )
            LOGGER.info(f"Ground truth actions: {ground_truth.actions}")
        except TextWorldEnvironmentError as e:
            LOGGER.warning(f"Ground truth not available: {e}")
        
        # Fallback: Run handcoded expert if ground truth unavailable
        expert_trajectory: Optional[ExpertTrajectory] = None
        if ground_truth is None:
            LOGGER.info("Falling back to handcoded expert baseline...")
            expert_trajectory = env.run_expert(max_steps=task_config.max_steps)
            LOGGER.info(
                f"Handcoded expert trajectory: {expert_trajectory.step_count} steps, "
                f"success={expert_trajectory.success}"
            )
            if not expert_trajectory.success:
                raise GreenAgentError(
                    f"ALFWorld handcoded expert failed to complete the task. "
                    f"Expert took {expert_trajectory.step_count} steps without success. "
                    f"Cannot evaluate white agent without a successful baseline trajectory."
                )
            # Reset environment for white agent
            reset_payload = env.reset_to_initial_state()
            observation = reset_payload["initial_observation"]
            LOGGER.info("Environment reset to initial state for white agent run")
        
        # Determine baseline actions (prefer ground truth)
        if ground_truth is not None:
            baseline_actions = ground_truth.actions
            baseline_source = "ground_truth"
            LOGGER.info(f"Using ground truth baseline: {len(baseline_actions)} steps")
        else:
            baseline_actions = expert_trajectory.actions
            baseline_source = "handcoded_expert"
            LOGGER.info(f"Using handcoded expert baseline: {len(baseline_actions)} steps")

        # Setup formatters for output
        demo_formatter = DemoFormatter() if self._demo and not benchmark_mode else None
        collecting_formatter = CollectingFormatter()
        output_parts: List[str] = []

        # Task Introduction (single log or demo output, not both)
        if demo_formatter:
            print(demo_formatter.format_task_intro(goal, observation), flush=True)
        else:
            LOGGER.info(f"Task: {goal[:150]}{'...' if len(goal) > 150 else ''}")

        # In demo mode, don't duplicate in response (already printed to stdout)
        if not demo_formatter:
            output_parts.append(
                collecting_formatter.format_task_intro(goal, observation)
            )
            output_parts.append(
                f"Episode '{env.episode_id}' started. Goal: {goal[:150]}"
            )

        # Create episode runner and execute
        runner = EpisodeRunner(env, self._messenger, formatter=demo_formatter)

        # Progress callback
        def _on_step_complete(
            step: int, action: str, reward: float, done: bool
        ) -> None:
            # In demo mode, step details are printed real-time; skip response duplication
            if not demo_formatter:
                output_parts.append(
                    collecting_formatter.format_step_result(
                        step,
                        action,
                        reward,
                        done,
                        env.metrics().get("success", False),
                        goal,
                    )
                )

        try:
            # Pass ground truth actions for oracle profile support
            # The white agent will only use them if configured as oracle
            gt_actions_for_oracle = (
                baseline_actions if baseline_source == "ground_truth" else None
            )
            
            episode_result = await runner.run_episode(
                white_agent_url=white_agent_url,
                goal=goal,
                initial_observation=observation,
                max_steps=task_config.max_steps,
                on_step_complete=_on_step_complete if not benchmark_mode else None,
                ground_truth_actions=gt_actions_for_oracle,
            )

            if episode_result.terminated_early:
                msg = f"Episode terminated early: {episode_result.termination_reason}"
                output_parts.append(msg)
                if demo_formatter:
                    print(demo_formatter.format_error(msg))

        except Exception as exc:
            LOGGER.error(f"Error during episode execution: {exc}", exc_info=True)
            raise

        # Post-episode evaluation using unified evaluate()
        LOGGER.info("Entering post-episode evaluation")
        trajectory = episode_result.trajectory
        success = episode_result.success
        step_count = episode_result.step_count
        LOGGER.info(f"Episode metrics - success: {success}, steps: {step_count}")

        # Use baseline trajectory for evaluation (ground truth preferred, fallback to handcoded)
        LOGGER.info(
            f"Using {baseline_source} baseline for evaluation: {len(baseline_actions)} steps"
        )

        # Use unified evaluation (combines heuristic + optional LLM)
        eval_result = await unified_evaluate(
            trajectory_steps=trajectory,
            goal=goal,
            success=success,
            task_id=str(task_config.task_index),
            step_budget=STEP_BUDGET,
            expert_actions=baseline_actions,
            expert_trajectory=expert_trajectory,  # May be None if using ground truth
            use_llm=True,
            baseline_source=baseline_source,  # Pass source for logging
        )

        # Format and output evaluation report
        if demo_formatter:
            # In demo mode, print detailed report to stdout (subprocess)
            # Only include eval_json in response (not full report) to avoid duplication
            print(demo_formatter.format_evaluation(eval_result, goal), flush=True)
            eval_json_payload = json.dumps(eval_result.to_dict())
            output_parts.append(f"<eval_json>{eval_json_payload}</eval_json>")
        else:
            # Non-demo mode: include full report in response
            outbound_report = collecting_formatter.format_evaluation(eval_result, goal)
            output_parts.append(outbound_report)

        LOGGER.info("Sending final report")
        combined_output = "\n\n".join(output_parts)
        await event_queue.enqueue_event(
            new_agent_text_message(combined_output, context_id=context.context_id)
        )

        LOGGER.info("Episode evaluation complete, resetting environment")
        env.reset()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(
            new_agent_text_message("Cancellation not implemented.")
        )


def start_green_agent(
    agent_name: str = "agent_card",
    *,
    host: str = "0.0.0.0",
    port: int = 9001,
) -> None:
    """Start the green agent HTTP service."""
    demo_mode = os.environ.get("DEMO_MODE", "0") == "1"
    level = (
        logging.DEBUG
        if os.environ.get("GREEN_VERBOSE") == "1"
        else (logging.WARNING if demo_mode else logging.INFO)
    )
    logging.basicConfig(level=level)
    if demo_mode:
        # Silence all logging; demo prints are handled via print()
        try:
            logging.disable(logging.CRITICAL)
        except Exception:
            pass
    # Suppress noisy third-party logs for cleaner output
    for noisy in (
        "httpx",
        "httpcore",
        "httpcore.connection",
        "httpcore.http11",
        "a2a.server.events.event_queue",  # suppress queue-closed warnings
        "a2a",
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
    ):
        try:
            # Use ERROR for event_queue, WARNING for others
            target_level = (
                logging.ERROR if noisy.endswith("event_queue") else logging.WARNING
            )
            logging.getLogger(noisy).setLevel(target_level)
        except Exception:
            pass
    LOGGER.info("Starting TextWorld green agent on %s:%s", host, port)

    agent_card_dict = load_agent_card_toml(agent_name)
    agent_card_dict["url"] = f"http://{host}:{port}"

    request_handler = DefaultRequestHandler(
        agent_executor=TextWorldGreenAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    application = A2AStarletteApplication(
        agent_card=AgentCard(**agent_card_dict),
        http_handler=request_handler,
    )

    import uvicorn

    uvicorn.run(
        application.build(), host=host, port=port, log_level="warning", access_log=False
    )


__all__ = ["TextWorldGreenAgentExecutor", "start_green_agent"]
