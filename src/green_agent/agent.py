"""Green agent implementation for TextWorld evaluation."""

from __future__ import annotations

import json
import logging
import re
import time
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional

import dotenv
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard
from a2a.utils import get_text_parts, new_agent_text_message
from litellm import completion

from ..utils.a2a_client import A2AMessenger
from ..utils.messaging import parse_tags, sanitize_action
from ..utils.textworld_env import (
    TaskConfig,
    TextWorldEnvironment,
    TextWorldEnvironmentError,
)
from .evaluator import LLMJudgeEvaluator, ScoreBreakdown, ReasoningTraceAssessment
from .rubric import load_rubric


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
        return tomllib.load(fh)


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
            self._evaluator = LLMJudgeEvaluator(self._rubric)
            LOGGER.info(
                f"Loaded evaluation rubric with {len(self._rubric.criteria)} criteria"
            )
        except Exception as exc:
            LOGGER.warning(
                f"Failed to load evaluation rubric: {exc}. Using fallback evaluation."
            )
            self._rubric = None
            self._evaluator = None

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Main execution loop for evaluation."""
        # Parse incoming task
        user_input = context.get_user_input()
        tags = parse_tags(user_input)

        if "white_agent_url" not in tags:
            raise GreenAgentError("Task payload missing <white_agent_url> tag")

        white_agent_url = tags["white_agent_url"]
        raw_task_config = tags.get("task_config") or tags.get("env_config") or "{}"
        try:
            task_config = TaskConfig.from_payload(json.loads(raw_task_config))
        except json.JSONDecodeError as exc:
            raise GreenAgentError("Invalid JSON in <task_config>") from exc

        # Setup environment
        env = TextWorldEnvironment(task_config)
        try:
            setup_payload = env.setup()
        except TextWorldEnvironmentError as exc:
            raise GreenAgentError(str(exc)) from exc

        goal = setup_payload["goal"]
        observation = setup_payload["initial_observation"]

        trajectory: List[Dict[str, Any]] = []

        await event_queue.enqueue_event(
            new_agent_text_message(
                f"Episode '{env.episode_id}' started. Goal: {goal[:150]}",
                context_id=context.context_id,
            )
        )

        # Interactive episode loop
        white_context_id = None
        current_observation = observation
        consecutive_failures = 0
        max_consecutive_failures = 3  # Fail episode after 3 consecutive failures

        LOGGER.info(f"Starting episode loop. Max steps: {task_config.max_steps}")

        try:
            for step in range(task_config.max_steps):
                LOGGER.info(f"Beginning step {step + 1}/{task_config.max_steps}")

                # Format message for white agent
                if step == 0:
                    message = self._format_initial_message(goal, observation)
                else:
                    message = self._format_observation_message(current_observation)

                # Log outgoing message to white agent
                LOGGER.info(
                    f"[Green→White] Step {step + 1}/{task_config.max_steps} - "
                    f"Sending message to white agent at {white_agent_url} "
                    f"(context_id={white_context_id})"
                )
                LOGGER.info(
                    f"[Green→White] Step {step + 1}/{task_config.max_steps} - "
                    f"Message content:\n{message}"
                )

                # Send to white agent and get command
                try:
                    white_response = await self._messenger.send_text(
                        white_agent_url,
                        message,
                        context_id=white_context_id,
                    )

                    if white_context_id is None:
                        white_context_id = white_response.context_id

                    # Extract full response text for logging
                    response_parts = get_text_parts(white_response.parts)
                    response_text = (
                        "\n".join(response_parts)
                        if response_parts
                        else "[empty response]"
                    )

                    # Log incoming message from white agent
                    LOGGER.info(
                        f"[White→Green] Step {step + 1}/{task_config.max_steps} - "
                        f"Received response from white agent "
                        f"(message_id={white_response.message_id}, "
                        f"context_id={white_response.context_id})"
                    )
                    LOGGER.info(
                        f"[White→Green] Step {step + 1}/{task_config.max_steps} - "
                        f"Response content:\n{response_text}"
                    )

                    reasoning, action = self._parse_command(white_response)

                    # Log reasoning and command separately
                    LOGGER.info(
                        f"[White→Green] Step {step + 1}/{task_config.max_steps} - "
                        f"Reasoning: {reasoning if reasoning else '(none)'}"
                    )
                    LOGGER.info(
                        f"[White→Green] Step {step + 1}/{task_config.max_steps} - "
                        f"Command: '{action}'"
                    )

                except Exception as exc:
                    consecutive_failures += 1
                    LOGGER.error(
                        f"White agent communication error (failure #{consecutive_failures}): {exc}",
                        exc_info=True,
                    )

                    # Fail episode if white agent is consistently unresponsive
                    if consecutive_failures >= max_consecutive_failures:
                        LOGGER.error(
                            f"White agent failed {consecutive_failures} consecutive times. "
                            f"Terminating episode early."
                        )
                        await event_queue.enqueue_event(
                            new_agent_text_message(
                                f"Episode terminated: White agent unresponsive after "
                                f"{consecutive_failures} consecutive failures.",
                                context_id=context.context_id,
                            )
                        )
                        break  # Exit the step loop early

                    reasoning = f"Error occurred during communication (failure #{consecutive_failures})"
                    action = "look"  # Fallback
                else:
                    # Reset failure counter on successful communication
                    if consecutive_failures > 0:
                        LOGGER.info(
                            f"White agent communication recovered after {consecutive_failures} failures"
                        )
                    consecutive_failures = 0

                # Execute action
                LOGGER.info(f"Executing action in environment: '{action}'")
                step_start_time = time.time()
                action_valid = True
                action_error = None

                try:
                    step_result = env.step(action)
                    LOGGER.info(
                        f"Step result - reward: {step_result.reward}, done: {step_result.done}"
                    )
                except TextWorldEnvironmentError as exc:
                    LOGGER.error(f"Environment error: {exc}", exc_info=True)
                    action_valid = False
                    action_error = str(exc)
                    await event_queue.enqueue_event(
                        new_agent_text_message(
                            f"Environment error: {exc}",
                            context_id=context.context_id,
                        )
                    )
                    break

                step_duration = time.time() - step_start_time

                # Detect observation changes
                prev_observation = current_observation
                new_observation = step_result.observation
                observation_changed = prev_observation != new_observation

                # Assess reasoning quality indicators
                reasoning_length = len(reasoning) if reasoning else 0
                reasoning_coherent = bool(reasoning and len(reasoning) > 10)

                # Store enhanced trajectory
                trajectory.append(
                    {
                        "step": step + 1,
                        "reasoning": reasoning,
                        "action": action,
                        "observation": step_result.observation,
                        "reward": step_result.reward,
                        "done": step_result.done,
                        "action_valid": action_valid,
                        "action_error": action_error,
                        "observation_changed": observation_changed,
                        "reasoning_length": reasoning_length,
                        "reasoning_coherent": reasoning_coherent,
                        "step_duration": step_duration,
                    }
                )

                current_observation = step_result.observation

                # Progress update
                await event_queue.enqueue_event(
                    new_agent_text_message(
                        f"Step {step + 1}: '{action}' → reward={step_result.reward:.2f}",
                        context_id=context.context_id,
                    )
                )

                if step_result.done:
                    LOGGER.info(f"Episode complete (done=True) after {step + 1} steps")
                    break

        finally:
            # Post-episode evaluation
            LOGGER.info("Entering post-episode evaluation")
            metrics = env.metrics()
            success = metrics.get("success", False)
            step_count = metrics.get("step_count", 0)
            LOGGER.info(f"Episode metrics - success: {success}, steps: {step_count}")

            # Evaluate trajectory using LLM judge
            if self._evaluator:
                LOGGER.info("Rating trajectory (quick)")
                rating_quick = await self._evaluator.rate_trajectory_quick(
                    goal, trajectory, success
                )
                LOGGER.info(f"Quick rating: {rating_quick}")

                LOGGER.info("Rating trajectory (detailed)")
                score_breakdown = await self._evaluator.rate_trajectory_detailed(
                    goal, trajectory, success
                )
                LOGGER.info(f"Detailed rating: {score_breakdown.overall_rating:.1f}")

                # Assess reasoning traces if enabled
                reasoning_assessment = None
                if self._rubric and self._rubric.reasoning_trace_analysis.get(
                    "enabled", False
                ):
                    LOGGER.info("Assessing reasoning traces")
                    reasoning_assessment = (
                        await self._evaluator.assess_reasoning_traces(goal, trajectory)
                    )
                    LOGGER.info("Reasoning trace assessment complete")

                # Generate enhanced report
                report = self._generate_enhanced_report(
                    success,
                    step_count,
                    task_config.max_steps,
                    rating_quick,
                    score_breakdown,
                    reasoning_assessment,
                )
            else:
                # Fallback to old evaluation method
                LOGGER.info("Using fallback evaluation (rubric not loaded)")
                rating_quick = await self._rate_trajectory_quick(
                    goal, trajectory, success
                )
                rating_detailed, reasoning = await self._rate_trajectory_detailed(
                    goal, trajectory, success
                )
                success_score = 1 if success else 0
                final_score = (
                    success_score + step_count + rating_quick + rating_detailed
                )

                emoji = "✅" if success else "❌"
                report = (
                    f"\nEvaluation Complete {emoji}\n"
                    f"{'=' * 50}\n"
                    f"Success: {success}\n"
                    f"Steps: {step_count} / {task_config.max_steps}\n"
                    f"Trajectory Ratings:\n"
                    f"  Quick Rating: {rating_quick:.1f}/10\n"
                    f"  Detailed Rating: {rating_detailed:.1f}/10\n"
                    f"  Reasoning: {reasoning[:300]}{'...' if len(reasoning) > 300 else ''}\n"
                    f"\nFinal Evaluation Score: {final_score:.1f}\n"
                    f"  (Success: {success_score} + Steps: {step_count} + "
                    f"Quick Rating: {rating_quick:.1f} + Detailed Rating: {rating_detailed:.1f})\n"
                )

            LOGGER.info("Sending final report")
            await event_queue.enqueue_event(
                new_agent_text_message(report, context_id=context.context_id)
            )

            LOGGER.info("Episode evaluation complete, resetting environment")
            env.reset()

    @staticmethod
    def _format_initial_message(goal: str, observation: str) -> str:
        """Format first message to white agent."""
        return f"""You are playing a TextWorld household task game.

<goal>
{goal}
</goal>

<observation>
{observation}
</observation>

<available_commands>
You can use the following command templates (replace placeholders with actual object/receptacle names):
- go to <recep>
- open <recep>
- close <recep>
- take <obj> from <recep>
- move <obj> to <recep>
- use <lamp>
- heat <obj> with <microwave>
- cool <obj> with <fridge>
- clean <obj> with <cleaner>
- slice <obj> with <knife>
- inventory
- look
- help
- examine <obj>
- examine <recep>
</available_commands>

Please respond with your chosen action wrapped in <command>...</command> tags.
Example: <command>go to kitchen</command>
"""

    @staticmethod
    def _format_observation_message(observation: str) -> str:
        """Format subsequent observation messages."""
        return f"""<observation>
{observation}
</observation>

<available_commands>
You can use the following command templates (replace placeholders with actual object/receptacle names):
- go to <recep>
- open <recep>
- close <recep>
- take <obj> from <recep>
- move <obj> to <recep>
- use <lamp>
- heat <obj> with <microwave>
- cool <obj> with <fridge>
- clean <obj> with <cleaner>
- slice <obj> with <knife>
- inventory
- look
- help
- examine <obj>
- examine <recep>
</available_commands>

What is your next command? Use <command>...</command> tags.
"""

    def _parse_command(self, white_response) -> tuple[str, str]:
        """Extract reasoning and command from white agent's response.

        Returns:
            Tuple of (reasoning: str, command: str)
        """
        message_parts = get_text_parts(white_response.parts)
        if not message_parts:
            return ("", "look")

        response_text = message_parts[0]
        tags = parse_tags(response_text)

        # Extract reasoning, default to empty string if missing
        reasoning = tags.get("reasoning", "").strip()

        # Extract command, default to "look" if missing
        if "command" in tags:
            command = sanitize_action(tags["command"])
        else:
            command = "look"

        return (reasoning, command)

    async def _rate_trajectory_quick(
        self, goal: str, trajectory: List[Dict[str, Any]], success: bool
    ) -> float:
        """Quick numeric rating without reasoning."""
        trajectory_text = self._format_trajectory(trajectory)

        prompt = f"""Rate this agent's performance from 1-10 (10=perfect).

Goal: {goal}
Success: {success}

Trajectory:
{trajectory_text}

Respond with ONLY a number. No explanation."""

        try:
            response = completion(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-4o",
                temperature=0.0,
            )

            rating_text = response.choices[0].message.content.strip()
            numbers = re.findall(r"\d+\.?\d*", rating_text)
            if numbers:
                rating = float(numbers[0])
                return max(1.0, min(10.0, rating))
        except Exception as exc:
            LOGGER.error(f"Rating failed: {exc}")

        # Fallback heuristic
        if success:
            efficiency = max(0, 1.0 - len(trajectory) / 50.0)
            return 7.0 + (3.0 * efficiency)
        return 3.0

    async def _rate_trajectory_detailed(
        self, goal: str, trajectory: List[Dict[str, Any]], success: bool
    ) -> tuple[float, str]:
        """Detailed rating with reasoning."""
        trajectory_text = self._format_trajectory(trajectory)

        prompt = f"""Evaluate this TextWorld agent's performance.

Goal: {goal}
Success: {success}

Trajectory:
{trajectory_text}

Provide:
1. Rating from 1-10
2. Detailed reasoning

Consider: efficiency, strategy quality, mistakes, success

Format:
Rating: [number]
Reasoning: [analysis]
"""

        try:
            response = completion(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-4o",
                temperature=0.3,
            )

            content = response.choices[0].message.content

            rating_match = re.search(r"Rating:\s*(\d+\.?\d*)", content)
            reasoning_match = re.search(r"Reasoning:\s*(.+)", content, re.DOTALL)

            if rating_match and reasoning_match:
                rating = float(rating_match.group(1))
                reasoning = reasoning_match.group(1).strip()
                return max(1.0, min(10.0, rating)), reasoning
        except Exception as exc:
            LOGGER.error(f"Detailed rating failed: {exc}")

        # Fallback
        if success:
            return 7.0, "Task completed successfully."
        return 3.0, "Task not completed."

    def _generate_enhanced_report(
        self,
        success: bool,
        step_count: int,
        max_steps: int,
        rating_quick: float,
        score_breakdown,
        reasoning_assessment,
    ) -> str:
        """Generate enhanced evaluation report with structured scores."""

        emoji = "✅" if success else "❌"
        report_lines = [
            f"\nEvaluation Complete {emoji}",
            "=" * 50,
            f"Success: {success}",
            f"Steps: {step_count} / {max_steps}",
            "",
            "Trajectory Ratings:",
            f"  Quick Rating: {rating_quick:.1f}/10",
        ]

        if isinstance(score_breakdown, ScoreBreakdown):
            overall_rating = score_breakdown.compute_weighted_overall()
            if overall_rating != score_breakdown.overall_rating:
                report_lines.append(
                    f"  Overall Rating (weighted): {overall_rating:.1f}/10"
                )
            else:
                report_lines.append(
                    f"  Overall Rating: {score_breakdown.overall_rating:.1f}/10"
                )

            # Category ratings
            if score_breakdown.category_ratings:
                report_lines.append("")
                report_lines.append("Per-Criterion Scores:")
                for key, cat_rating in score_breakdown.category_ratings.items():
                    criterion_name = cat_rating.criterion_name
                    report_lines.append(
                        f"  {criterion_name}: {cat_rating.score:.1f}/10 "
                        f"(weight: {cat_rating.weight:.0%}, "
                        f"weighted: {cat_rating.weighted_score():.2f})"
                    )

            # Qualitative assessments
            if score_breakdown.qualitative_assessments:
                report_lines.append("")
                report_lines.append("Qualitative Assessment:")

                if "strengths" in score_breakdown.qualitative_assessments:
                    report_lines.append("  Strengths:")
                    for strength in score_breakdown.qualitative_assessments[
                        "strengths"
                    ]:
                        report_lines.append(f"    - {strength}")

                if "weaknesses" in score_breakdown.qualitative_assessments:
                    report_lines.append("  Weaknesses:")
                    for weakness in score_breakdown.qualitative_assessments[
                        "weaknesses"
                    ]:
                        report_lines.append(f"    - {weakness}")

                if "notable_behaviors" in score_breakdown.qualitative_assessments:
                    report_lines.append("  Notable Behaviors:")
                    for behavior in score_breakdown.qualitative_assessments[
                        "notable_behaviors"
                    ]:
                        report_lines.append(f"    - {behavior}")

                if "recommendations" in score_breakdown.qualitative_assessments:
                    report_lines.append("  Recommendations:")
                    for rec in score_breakdown.qualitative_assessments[
                        "recommendations"
                    ]:
                        report_lines.append(f"    - {rec}")

            # Detailed reasoning
            if score_breakdown.detailed_reasoning:
                report_lines.append("")
                report_lines.append("Detailed Reasoning:")
                reasoning_text = score_breakdown.detailed_reasoning[:500]
                if len(score_breakdown.detailed_reasoning) > 500:
                    reasoning_text += "..."
                report_lines.append(f"  {reasoning_text}")

        # Reasoning trace analysis
        if isinstance(reasoning_assessment, ReasoningTraceAssessment):
            report_lines.append("")
            report_lines.append("Reasoning Trace Analysis:")
            report_lines.append(
                f"  Reasoning Quality: {reasoning_assessment.reasoning_quality}"
            )
            report_lines.append(
                f"  Planning Evidence: {reasoning_assessment.planning_evidence}"
            )
            report_lines.append(
                f"  Error Handling: {reasoning_assessment.error_handling}"
            )

        return "\n".join(report_lines)

    @staticmethod
    def _format_trajectory(trajectory: List[Dict[str, Any]]) -> str:
        """Format trajectory for LLM rating (enhanced with reasoning traces)."""
        lines = []
        for step_data in trajectory:
            step_num = step_data.get("step", "?")
            reasoning = step_data.get("reasoning", "")
            action = step_data.get("action", "")
            reward = step_data.get("reward", 0.0)

            # Enhanced formatting to highlight reasoning traces
            if reasoning:
                reasoning_snippet = (
                    reasoning[:150] + "..." if len(reasoning) > 150 else reasoning
                )
                lines.append(
                    f"Step {step_num}: [Reasoning: {reasoning_snippet}] → Action: {action} "
                    f"(reward: {reward:.2f})"
                )
            else:
                lines.append(
                    f"Step {step_num}: [No reasoning provided] → Action: {action} "
                    f"(reward: {reward:.2f})"
                )

            # Add observation context if available and changed
            if step_data.get("observation_changed", False):
                obs = step_data.get("observation", "")
                if obs:
                    obs_snippet = obs[:80].replace("\n", " ")
                    lines.append(
                        f"  → Observation: {obs_snippet}{'...' if len(obs) > 80 else ''}"
                    )
        return "\n".join(lines)

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
    logging.basicConfig(level=logging.INFO)
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

    uvicorn.run(application.build(), host=host, port=port)


__all__ = ["TextWorldGreenAgentExecutor", "start_green_agent"]
