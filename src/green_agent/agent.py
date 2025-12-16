"""Green agent implementation for TextWorld evaluation."""

from __future__ import annotations

import json
import logging
import re
import time

try:
    import tomllib as tomli  # Python 3.11+
except ModuleNotFoundError:
    import tomli  # Backport for <=3.10
import os
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

from ..utils.a2a_client import A2AMessenger
from ..utils.vllm_client import completion as vllm_completion
from ..utils.messaging import parse_tags, sanitize_action
from ..utils.textworld_env import (
    TaskConfig,
    TextWorldEnvironment,
    TextWorldEnvironmentError,
)
from .green_assessor import (
    STEP_BUDGET,
    TrajectoryEval,
    evaluate_trajectory,
    print_task_eval,
    compute_weighted_overall,
)
from .evaluator import (
    LLMJudgeEvaluator,
    ScoreBreakdown,
    ReasoningTraceAssessment,
    evaluate_with_llm,
)
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
        # Demo mode: only print human-friendly green/white exchanges
        self._demo = os.environ.get("DEMO_MODE", "0") == "1"

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Main execution loop for evaluation."""
        # Parse incoming task
        user_input = context.get_user_input()
        tags = parse_tags(user_input)
        fast_white = tags.get("fast_white", "0").strip() == "1"

        white_agent_url = tags.get("white_agent_url")
        if not fast_white and not white_agent_url:
            raise GreenAgentError("Task payload missing <white_agent_url> tag")
        raw_task_config = tags.get("task_config") or tags.get("env_config") or "{}"
        benchmark_mode = tags.get("benchmark_mode", "0").strip() == "1"
        try:
            task_config = TaskConfig.from_payload(json.loads(raw_task_config))
        except json.JSONDecodeError as exc:
            raise GreenAgentError("Invalid JSON in <task_config>") from exc

        # Setup environment
        env = TextWorldEnvironment(task_config, use_expert_plan=fast_white)
        try:
            setup_payload = env.setup()
        except TextWorldEnvironmentError as exc:
            raise GreenAgentError(str(exc)) from exc

        goal = setup_payload["goal"]
        observation = setup_payload["initial_observation"]

        trajectory: List[Dict[str, Any]] = []

        # Capture a specific task string from the initial observation if present
        extracted_task_text = None
        try:
            _m = re.search(r"Your task is to:\s*(.+)", observation, flags=re.IGNORECASE)
            if _m:
                extracted_task_text = _m.group(1).strip()
        except Exception:
            extracted_task_text = None

        # Accumulate output for single response
        output_parts: List[str] = []

        def _collect_output(text: str) -> None:
            """Collect output for final response."""
            output_parts.append(text)

        # Legacy helper for compatibility (now just collects output)
        async def _safe_enqueue(text: str) -> None:
            _collect_output(text)

        # Task Introduction (concise)
        actions_hint = "- go to <recep>, open/close <recep>, take/move <obj>, inventory, look, examine"
        if self._demo and not benchmark_mode:
            print("— Task Introduction —")
            print(f"- What is the task? {goal}")
            print("- What does the environment look like?")
            print(observation)
            print(f"- What actions can each agent take? {actions_hint}")
        else:
            LOGGER.info("Task Introduction:")
            LOGGER.info(
                " - What is the task? %s",
                (goal[:200] + ("..." if len(goal) > 200 else "")),
            )
            LOGGER.info(
                " - What does the environment look like? %s",
                (
                    observation[:200].replace("\n", " ")
                    + ("..." if len(observation) > 200 else "")
                ),
            )
            LOGGER.info(" - What actions can the white agent take? %s", actions_hint)

        if not fast_white:
            _collect_output(
                "Task Introduction\n"
                f"- What is the task? {goal}\n"
                f"- What does the environment look like? {observation[:300]}{'...' if len(observation) > 300 else ''}\n"
                f"- What actions can each agent take? {actions_hint}"
            )
            _collect_output(f"Episode '{env.episode_id}' started. Goal: {goal[:150]}")

        # Interactive episode loop
        white_context_id = None
        current_observation = observation
        consecutive_failures = 0
        max_consecutive_failures = 3  # Fail episode after 3 consecutive failures

        LOGGER.info(f"Starting episode loop. Max steps: {task_config.max_steps}")

        try:
            if fast_white:
                trajectory.extend(
                    await self._run_fast_mode(
                        env=env,
                        event_queue=event_queue,
                        context=context,
                        goal=goal,
                        initial_observation=observation,
                        max_steps=task_config.max_steps,
                        benchmark_mode=benchmark_mode,
                        extracted_task_text=extracted_task_text,
                        _safe_enqueue=_safe_enqueue,
                    )
                )
            else:
                for step in range(task_config.max_steps):
                    LOGGER.info(f"Step {step + 1}/{task_config.max_steps} - begin")

                    # Format message for white agent
                    if step == 0:
                        message = self._format_initial_message(goal, observation)
                    else:
                        message = self._format_observation_message(current_observation)

                    # Log outgoing message to white agent
                    if not self._demo or benchmark_mode:
                        LOGGER.info(
                            f"[Green→White] Step {step + 1}/{task_config.max_steps} - "
                            f"Sending message to white agent at {white_agent_url} "
                            f"(context_id={white_context_id})"
                        )
                        LOGGER.debug(
                            f"[Green→White] Step {step + 1}/{task_config.max_steps} - "
                            f"Message content:\n{message}"
                        )
                    else:
                        # Human-readable green output
                        if step == 0:
                            # Styling
                            BOLD = "\033[1m"
                            # Prefer 256-color bright green; falls back fine on most terminals
                            GREEN_FG = "\033[38;5;46m"
                            RESET = "\033[0m"
                            DIV = "-" * 80
                            step_header = f"{BOLD}Step {step + 1}{RESET}"
                            print(DIV)
                            print(
                                ""
                            )  # single blank line between divider and step header
                            print(step_header)
                            print("")  # blank line after step header
                            print(f"{BOLD}{GREEN_FG}GREEN{RESET}:")
                            print(f"Observation: {BOLD}{observation}{RESET}")
                            print("")  # spacing after green block
                        else:
                            BOLD = "\033[1m"
                            GREEN_FG = "\033[38;5;46m"
                            RESET = "\033[0m"
                            DIV = "-" * 80
                            step_header = f"{BOLD}Step {step + 1}{RESET}"
                            print(DIV)
                            print(
                                ""
                            )  # single blank line between divider and step header
                            print(step_header)
                            print("")  # blank line after step header
                            print(f"{BOLD}{GREEN_FG}GREEN{RESET}:")
                            print(f"Observation: {BOLD}{current_observation}{RESET}")
                            print("")  # spacing after green block

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
                        if not self._demo or benchmark_mode:
                            LOGGER.info(
                                f"[White→Green] Step {step + 1}/{task_config.max_steps} - "
                                f"Received response from white agent "
                                f"(message_id={white_response.message_id}, "
                                f"context_id={white_response.context_id})"
                            )
                            LOGGER.debug(
                                f"[White→Green] Step {step + 1}/{task_config.max_steps} - "
                                f"Response content:\n{response_text}"
                            )

                        reasoning, action = self._parse_command(white_response)

                        if self._demo:
                            BOLD = "\033[1m"
                            RESET = "\033[0m"
                            print(f"{BOLD}WHITE{RESET}:")
                            if reasoning:
                                _rs = reasoning[:300].replace("\n", " ")
                                print(
                                    f"Reasoning: {_rs}{'...' if len(reasoning) > 300 else ''}"
                                )
                            print(f"Command: {BOLD}{action}{RESET}")
                        else:
                            # Log reasoning and command separately
                            LOGGER.debug(
                                f"[White→Green] Step {step + 1}/{task_config.max_steps} - "
                                f"Reasoning: {reasoning if reasoning else '(none)'}"
                            )
                            LOGGER.info(
                                f"[White→Green] Step {step + 1}/{task_config.max_steps} - "
                                f"Command: '{action}'"
                            )

                    except Exception as exc:
                        consecutive_failures += 1
                        if self._demo:
                            # Quiet, human-friendly note in demo mode
                            print(
                                "Note: temporary communication issue with White agent. Retrying..."
                            )
                        else:
                            LOGGER.error(
                                f"White agent communication error (failure #{consecutive_failures}): {exc}",
                                exc_info=True,
                            )

                        # Fail episode if white agent is consistently unresponsive
                        if consecutive_failures >= max_consecutive_failures:
                            if self._demo:
                                print(
                                    "White agent became unavailable. Ending episode early."
                                )
                            else:
                                LOGGER.error(
                                    f"White agent failed {consecutive_failures} consecutive times. "
                                    f"Terminating episode early."
                                )
                            _collect_output(
                                f"Episode terminated: White agent unresponsive after "
                                f"{consecutive_failures} consecutive failures."
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
                        await _safe_enqueue(f"Environment error: {exc}")
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
                    if not fast_white:
                        await _safe_enqueue(
                            f"Step {step + 1}: '{action}' → reward={step_result.reward:.2f}, done={step_result.done}"
                        )
                        # Concise per-step summary at INFO for console
                        if not self._demo or benchmark_mode:
                            LOGGER.info(
                                f"Step {step + 1}: action='{action}', valid={action_valid}, "
                                f"reward={step_result.reward:.2f}, done={step_result.done}, "
                                f"obs_changed={observation_changed}"
                            )
                        else:
                            # Human-readable per-step outcome and task status
                            success_now = bool(env.metrics().get("success", False))
                            status_str = (
                                "SUCCESSFUL ✅" if success_now else "unsuccessful ❌"
                            )
                            print("")  # spacing before status
                            # Prefer task text from the observation if present: "Your task is to: ..."
                            _obs_for_task = (
                                current_observation
                                if current_observation
                                else observation
                            )
                            _task_match = re.search(
                                r"Your task is to:\s*(.+)",
                                _obs_for_task,
                                flags=re.IGNORECASE,
                            )
                            if _task_match:
                                _task_text = _task_match.group(1).strip()
                            elif extracted_task_text:
                                _task_text = extracted_task_text
                            else:
                                _task_text = goal
                            print(f"Task: {_task_text}")
                            print(f"Status: {status_str}")
                            print("")  # spacing after status

                        if step_result.done:
                            LOGGER.info(
                                f"Episode complete (done=True) after {step + 1} steps"
                            )
                            break

        except Exception as exc:
            LOGGER.error(f"Error during episode execution: {exc}", exc_info=True)
            raise

        # Post-episode evaluation using deterministic rubric (must run before finally)
        LOGGER.info("Entering post-episode evaluation")
        metrics = env.metrics()
        success = metrics.get("success", False)
        step_count = metrics.get("step_count", 0)
        LOGGER.info(f"Episode metrics - success: {success}, steps: {step_count}")

        # Try LLM evaluation first for reasoning/strategy (semantic assessment)
        llm_scores = None
        try:
            llm_scores = await evaluate_with_llm(
                goal=goal,
                trajectory=trajectory,
                success=success,
                step_budget=task_config.max_steps,
            )
        except Exception as e:
            LOGGER.warning(f"LLM evaluation failed: {e}")

        # Get heuristic evaluation (always needed for correctness/efficiency and as fallback)
        heuristic_eval: TrajectoryEval = evaluate_trajectory(
            task_id=str(task_config.task_index),
            task_text=goal,
            step_budget=STEP_BUDGET,
            trajectory_steps=trajectory,
            env_success=success,
        )

        # Merge: use LLM scores for reasoning/strategy if available
        if llm_scores:
            LOGGER.info(
                f"Using LLM evaluation: reasoning={llm_scores['reasoning_score']:.1f}, "
                f"strategy={llm_scores['strategy_score']:.1f}"
            )
            eval_result = TrajectoryEval(
                task_id=heuristic_eval.task_id,
                success=heuristic_eval.success,
                steps=heuristic_eval.steps,
                correctness=heuristic_eval.correctness,
                efficiency=heuristic_eval.efficiency,
                strategy=round(llm_scores["strategy_score"], 2),
                reasoning=round(llm_scores["reasoning_score"], 2),
                overall=compute_weighted_overall(
                    heuristic_eval.correctness,
                    heuristic_eval.efficiency,
                    llm_scores["strategy_score"],
                    llm_scores["reasoning_score"],
                ),
                quick=heuristic_eval.quick,
                features={
                    **heuristic_eval.features,
                    "llm_evaluated": True,
                },
                notes={
                    **(
                        heuristic_eval.notes
                        if isinstance(heuristic_eval.notes, dict)
                        else {}
                    ),
                    "reasoning_rationale": llm_scores["reasoning_rationale"],
                    "strategy_rationale": llm_scores["strategy_rationale"],
                    "notable_moments": llm_scores.get("notable_moments", []),
                },
            )
        else:
            LOGGER.info("Using heuristic evaluation (LLM unavailable)")
            eval_result = heuristic_eval

        report = print_task_eval(eval_result, task_text=goal, step_budget=STEP_BUDGET)
        eval_json_payload = json.dumps(eval_result.to_dict())
        eval_tag = f"<eval_json>{eval_json_payload}</eval_json>"
        outbound_report = f"{report}\n{eval_tag}"

        if self._demo:
            print("\n— Evaluation —\n")
            print(report)

        # Add evaluation report to output
        _collect_output(outbound_report)

        LOGGER.info("Sending final report")
        # Combine all output parts into single response
        combined_output = "\n\n".join(output_parts)
        await event_queue.enqueue_event(
            new_agent_text_message(combined_output, context_id=context.context_id)
        )

        LOGGER.info("Episode evaluation complete, resetting environment")
        env.reset()

    @staticmethod
    def _analyze_trajectory_for_strategy(
        trajectory: List[Dict[str, Any]], goal: str
    ) -> str:
        """Heuristic strategy analysis to produce a longer narrative."""
        total_steps = len(trajectory)
        if total_steps == 0:
            return "No actions taken; strategy cannot be assessed."
        actions = [str(x.get("action", "")).strip() for x in trajectory]
        # Count navigation, interactions, and revisits of 'go to'
        go_to_targets: list[str] = []
        for a in actions:
            m = re.match(r"^go to (.+)$", a)
            if m:
                go_to_targets.append(m.group(1))
        unique_nav = len(set(go_to_targets)) if go_to_targets else 0
        revisits = max(0, len(go_to_targets) - unique_nav) if go_to_targets else 0
        open_actions = sum(1 for a in actions if a.startswith("open "))
        take_actions = sum(1 for a in actions if a.startswith("take "))
        put_actions = sum(
            1 for a in actions if a.startswith("put ") or a.startswith("move ")
        )
        examine_actions = sum(
            1 for a in actions if a.startswith("examine") or a == "look"
        )
        # Observation change rate as a proxy for progress/adaptation
        changed = sum(1 for x in trajectory if x.get("observation_changed", False))
        change_rate = changed / float(total_steps)
        # Compose
        parts: list[str] = []
        parts.append(
            f"The trajectory contains {total_steps} actions with {unique_nav} unique navigation targets and {revisits} revisits."
        )
        parts.append(
            f"It balances navigation with interactions (open={open_actions}, take={take_actions}, put/move={put_actions})."
        )
        if examine_actions > 0:
            parts.append(
                f"Exploration uses inspections ({examine_actions} examine/look actions) to gather state before acting."
            )
        if change_rate >= 0.6:
            parts.append(
                "Most actions lead to state changes, suggesting purposeful progression rather than random wandering."
            )
        elif change_rate >= 0.3:
            parts.append(
                "A moderate portion of actions affect the environment, indicating some focused progression with detours."
            )
        else:
            parts.append(
                "Few actions change the environment, indicating inefficiency or repeated non-progressing actions."
            )
        parts.append(
            "Overall, the sequence appears "
            + (
                "well prioritized and adaptive"
                if change_rate >= 0.6
                else "somewhat systematic but inconsistently prioritized"
                if change_rate >= 0.3
                else "loosely organized with limited prioritization"
            )
        )
        return " ".join(parts)

    @staticmethod
    def _analyze_trajectory_for_reasoning(
        trajectory: List[Dict[str, Any]], goal: str
    ) -> str:
        """Heuristic reasoning-quality analysis to produce a longer narrative."""
        if not trajectory:
            return "No reasoning provided."
        reasonings = [
            str(x.get("reasoning", "")).strip()
            for x in trajectory
            if str(x.get("reasoning", "")).strip()
        ]
        coverage = len(reasonings) / float(len(trajectory))
        mentions_goal = sum(
            1
            for r in reasonings
            if any(tok in r.lower() for tok in ["goal", "task", "book", "salt", "lamp"])
        )  # coarse proxy
        coherence = (
            sum(1 for r in reasonings if len(r) >= 20) / float(len(reasonings))
            if reasonings
            else 0.0
        )
        parts: list[str] = []
        parts.append(
            f"Reasoning coverage is {coverage:.0%} of steps; longer explanations occur in {coherence:.0%} of provided entries."
        )
        if mentions_goal / float(len(reasonings) or 1) >= 0.5:
            parts.append(
                "Many entries reference the goal or task objects directly, showing goal awareness."
            )
        else:
            parts.append(
                "Few entries reference the goal explicitly; explanations trend generic."
            )
        parts.append(
            "Overall, the reasoning is "
            + (
                "coherent and grounded with frequent goal references"
                if coherence >= 0.6 and coverage >= 0.6
                else "partly coherent with some grounding but intermittently generic"
                if coherence >= 0.4
                else "brief and generic, with limited grounding in observations"
            )
        )
        return " ".join(parts)

    @staticmethod
    def _generate_demo_evaluation_report_v2(
        goal: str,
        *,
        success: bool,
        step_count: int,
        max_steps: int,
        score_breakdown,
        trajectory: List[Dict[str, Any]],
    ) -> str:
        """Demo-friendly evaluation block with longer Strategy/Reasoning narratives."""

        def _cat_score(key: str, default: float | None = None) -> float | None:
            try:
                cr = (
                    score_breakdown.category_ratings.get(key)
                    if score_breakdown and score_breakdown.category_ratings
                    else None
                )
                return float(cr.score) if cr else default
            except Exception:
                return default

        correctness_score = 10.0 if success else 0.0
        correctness_reasoning = (
            "Env reports success (goal state reached)."
            if success
            else "Env reports failure (goal state not reached)."
        )
        correctness_calc = f"success == {'True' if success else 'False'} → score = {correctness_score:.1f} / 10"

        ratio = (step_count / float(max_steps)) if max_steps > 0 else 1.0
        ratio = max(0.0, min(1.0, ratio))
        efficiency_score = max(0.0, 10.0 * (1.0 - ratio))
        efficiency_reasoning = f"Uses {step_count} of {max_steps} budgeted steps → " + (
            "efficient path."
            if ratio < 0.4
            else "moderate wandering."
            if ratio < 0.7
            else "heavy wandering."
        )
        efficiency_calc = f"ratio = {step_count} / {max_steps} = {ratio:.2f}\n    score = 10 × (1 − {ratio:.2f}) = {efficiency_score:.1f} / 10"

        strategy_score = _cat_score("strategy_quality", 5.0) or 5.0
        strategy_reasoning = (
            TextWorldGreenAgentExecutor._analyze_trajectory_for_strategy(
                trajectory, goal
            )
        )

        reasoning_quality_score = _cat_score("reasoning_quality", 5.0) or 5.0
        reasoning_quality_reasoning = (
            TextWorldGreenAgentExecutor._analyze_trajectory_for_reasoning(
                trajectory, goal
            )
        )

        weighted_overall = None
        try:
            if score_breakdown:
                weighted_overall = float(score_breakdown.compute_weighted_overall())
        except Exception:
            weighted_overall = None
        if weighted_overall is None:
            weighted_overall = (
                correctness_score
                + efficiency_score
                + strategy_score
                + reasoning_quality_score
            ) / 4.0

        lines: list[str] = []
        lines.append("--- GREEN AGENT EVALUATION ---")
        lines.append("")
        lines.append(f"Task Success: {'YES' if success else 'NO'}")
        lines.append(f"Steps Used: {step_count} / {max_steps}")
        lines.append("")
        lines.append("Correctness / Task Completion")
        lines.append(f"  Reasoning: {correctness_reasoning}")
        lines.append(f"  Calculation: {correctness_calc}")
        lines.append("")
        lines.append("Efficiency (Step Budget)")
        lines.append(f"  Reasoning: {efficiency_reasoning}")
        lines.append("  Calculation:")
        lines.append(f"    {efficiency_calc}")
        lines.append("")
        lines.append("Strategy Quality")
        lines.append(f"  Reasoning: {strategy_reasoning}")
        lines.append(f"  Score: {strategy_score:.1f} / 10   (rubric-based)")
        lines.append("")
        lines.append("Reasoning Quality")
        lines.append(f"  Reasoning: {reasoning_quality_reasoning}")
        lines.append(f"  Score: {reasoning_quality_score:.1f} / 10   (rubric-based)")
        lines.append("")
        lines.append(f"Overall Rating (weighted): {weighted_overall:.1f} / 10")
        return "\n".join(lines)

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

    async def _run_fast_mode(
        self,
        *,
        env: TextWorldEnvironment,
        event_queue: EventQueue,
        context: RequestContext,
        goal: str,
        initial_observation: str,
        max_steps: int,
        benchmark_mode: bool,
        extracted_task_text: Optional[str],
        _safe_enqueue,
    ) -> List[Dict[str, Any]]:
        """Execute expert-plan actions without calling the LLM white agent."""
        LOGGER.info("Fast white mode enabled: executing expert plan locally.")
        trajectory: List[Dict[str, Any]] = []
        current_observation = initial_observation
        step_idx = 0

        while step_idx < max_steps:
            plan = env.walkthrough()
            if not plan:
                LOGGER.warning(
                    "Fast white mode requested but no expert plan available."
                )
                await _safe_enqueue(
                    "Fast white mode: no expert plan available for this task. Ending episode early."
                )
                break

            action = plan[0]
            LOGGER.info(
                f"[FastWhite] Step {step_idx + 1}/{max_steps} - executing expert action '{action}'"
            )
            step_start = time.time()
            action_valid = True
            action_error = None

            try:
                step_result = env.step(action)
            except TextWorldEnvironmentError as exc:
                LOGGER.error(
                    f"Environment error during fast mode: {exc}", exc_info=True
                )
                action_valid = False
                action_error = str(exc)
                await _safe_enqueue(f"Environment error: {exc}")
                break

            step_duration = time.time() - step_start
            observation_changed = current_observation != step_result.observation
            reasoning = "fast_white (expert plan)"
            trajectory.append(
                {
                    "step": step_idx + 1,
                    "reasoning": reasoning,
                    "action": action,
                    "observation": step_result.observation,
                    "reward": step_result.reward,
                    "done": step_result.done,
                    "action_valid": action_valid,
                    "action_error": action_error,
                    "observation_changed": observation_changed,
                    "reasoning_length": len(reasoning),
                    "reasoning_coherent": True,
                    "step_duration": step_duration,
                }
            )

            current_observation = step_result.observation
            await _safe_enqueue(
                f"Step {step_idx + 1}: '{action}' → reward={step_result.reward:.2f}, done={step_result.done}"
            )

            if not self._demo or benchmark_mode:
                LOGGER.info(
                    f"[FastWhite] Step {step_idx + 1}: action='{action}', reward={step_result.reward:.2f}, "
                    f"done={step_result.done}, obs_changed={observation_changed}"
                )
            else:
                success_now = bool(env.metrics().get("success", False))
                status_str = "SUCCESSFUL ✅" if success_now else "unsuccessful ❌"
                print("")
                _obs_for_task = (
                    current_observation if current_observation else initial_observation
                )
                _task_match = re.search(
                    r"Your task is to:\s*(.+)", _obs_for_task, flags=re.IGNORECASE
                )
                if _task_match:
                    _task_text = _task_match.group(1).strip()
                elif extracted_task_text:
                    _task_text = extracted_task_text
                else:
                    _task_text = goal
                print(f"Task: {_task_text}")
                print(f"Status: {status_str}")
                print("")

            if step_result.done or env.metrics().get("success", False):
                LOGGER.info(
                    f"[FastWhite] Episode complete after {step_idx + 1} steps "
                    f"(done={step_result.done}, success={env.metrics().get('success', False)})"
                )
                break

            step_idx += 1

        return trajectory

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
            response = vllm_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )

            rating_text = response.content.strip()
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
            response = vllm_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            content = response.content

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
    # Suppress noisy third-party logs for demo-friendly output
    for noisy in (
        "httpx",
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
