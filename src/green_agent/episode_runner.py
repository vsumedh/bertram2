"""Episode execution logic for TextWorld evaluation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from a2a.utils import get_text_parts

from ..utils.a2a_client import A2AMessenger
from ..utils.messaging import parse_tags, sanitize_action
from ..utils.textworld_env import TextWorldEnvironment, TextWorldEnvironmentError

if TYPE_CHECKING:
    from .output_formatter import OutputFormatter

LOGGER = logging.getLogger(__name__)


@dataclass
class TrajectoryStep:
    """Single step in episode trajectory."""

    step: int
    reasoning: str
    action: str
    observation: str
    reward: float
    done: bool
    action_valid: bool = True
    action_error: Optional[str] = None
    observation_changed: bool = False
    reasoning_length: int = 0
    reasoning_coherent: bool = False
    step_duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for evaluation."""
        return {
            "step": self.step,
            "reasoning": self.reasoning,
            "action": self.action,
            "observation": self.observation,
            "reward": self.reward,
            "done": self.done,
            "action_valid": self.action_valid,
            "action_error": self.action_error,
            "observation_changed": self.observation_changed,
            "reasoning_length": self.reasoning_length,
            "reasoning_coherent": self.reasoning_coherent,
            "step_duration": self.step_duration,
        }


@dataclass
class EpisodeResult:
    """Result of running an episode."""

    trajectory: List[Dict[str, Any]]
    goal: str
    success: bool
    step_count: int
    expert_plan: Optional[List[str]] = None
    terminated_early: bool = False
    termination_reason: Optional[str] = None


class EpisodeRunner:
    """Runs TextWorld episodes, collecting trajectory data."""

    def __init__(
        self,
        env: TextWorldEnvironment,
        messenger: A2AMessenger,
        *,
        max_consecutive_failures: int = 3,
        formatter: Optional["OutputFormatter"] = None,
    ):
        """Initialize episode runner.

        Args:
            env: TextWorld environment instance
            messenger: A2A messenger for white agent communication
            max_consecutive_failures: Fail episode after this many consecutive failures
            formatter: Optional output formatter for demo/logging mode
        """
        self._env = env
        self._messenger = messenger
        self._max_consecutive_failures = max_consecutive_failures
        self._formatter = formatter

    async def run_episode(
        self,
        white_agent_url: str,
        goal: str,
        initial_observation: str,
        max_steps: int,
        *,
        on_step_complete: Optional[Callable[[int, str, float, bool], None]] = None,
        ground_truth_actions: Optional[List[str]] = None,
    ) -> EpisodeResult:
        """Run episode with white agent, return trajectory.

        Args:
            white_agent_url: URL of the white agent
            goal: Task goal description
            initial_observation: Initial environment observation
            max_steps: Maximum steps allowed
            on_step_complete: Optional callback(step, action, reward, done) for progress
            ground_truth_actions: Optional ground truth actions for oracle profile

        Returns:
            EpisodeResult with trajectory and metadata
        """
        trajectory: List[Dict[str, Any]] = []
        white_context_id = None
        current_observation = initial_observation
        consecutive_failures = 0
        terminated_early = False
        termination_reason = None
        # Get initial admissible commands from environment
        current_admissible = self._env.admissible_commands

        LOGGER.info(f"Starting episode loop. Max steps: {max_steps}")
        if ground_truth_actions:
            LOGGER.info(f"Ground truth actions provided: {len(ground_truth_actions)} steps")

        for step in range(max_steps):
            # Format message for white agent with goal and valid actions
            if step == 0:
                message = self._format_initial_message(
                    goal, initial_observation, current_admissible,
                    ground_truth_actions=ground_truth_actions,
                )
            else:
                message = self._format_observation_message(
                    goal, current_observation, current_admissible
                )

            obs_to_display = current_observation if step > 0 else initial_observation

            # Consolidated step start log (single line for key info)
            LOGGER.info(
                f"[Step {step + 1}/{max_steps}] Green→White: obs_len={len(obs_to_display)}, "
                f"valid_actions={len(current_admissible)}"
            )
            # Full details at DEBUG level
            LOGGER.debug(f"  Observation: {obs_to_display[:200]}...")
            LOGGER.debug(f"  Valid actions: {current_admissible[:5]}...")
            LOGGER.debug(f"  Full message:\n{message}")

            # Demo mode: print formatted step start (GREEN agent's turn)
            if self._formatter:
                step_start_output = self._formatter.format_step_start(
                    step + 1, max_steps, obs_to_display
                )
                if step_start_output:
                    print(step_start_output, flush=True)

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
                    "\n".join(response_parts) if response_parts else "[empty response]"
                )

                reasoning, action = self._parse_command(white_response)

                # Consolidated white response log (single line for key info)
                reasoning_preview = (
                    (reasoning[:80].replace("\n", " ") + "...")
                    if reasoning and len(reasoning) > 80
                    else (reasoning or "(none)")
                )
                LOGGER.info(
                    f"[Step {step + 1}/{max_steps}] White→Green: action='{action}', "
                    f"reasoning_len={len(reasoning) if reasoning else 0}"
                )
                LOGGER.debug(f"  Reasoning: {reasoning_preview}")
                LOGGER.debug(f"  Full response:\n{response_text}")

                # Demo mode: print formatted white agent response
                if self._formatter:
                    white_output = self._formatter.format_white_response(
                        reasoning, action
                    )
                    if white_output:
                        print(white_output, flush=True)

                # Validate and correct the action against admissible commands
                action, was_valid = self._validate_and_correct_action(
                    action, current_admissible
                )
                if not was_valid:
                    LOGGER.warning(
                        f"[White→Green] Step {step + 1}/{max_steps} - "
                        f"Action corrected to: '{action}'"
                    )

            except Exception as exc:
                consecutive_failures += 1
                LOGGER.error(
                    f"White agent communication error (failure #{consecutive_failures}): {exc}",
                    exc_info=True,
                )

                # Fail episode if white agent is consistently unresponsive
                if consecutive_failures >= self._max_consecutive_failures:
                    LOGGER.error(
                        f"White agent failed {consecutive_failures} consecutive times. "
                        f"Terminating episode early."
                    )
                    terminated_early = True
                    termination_reason = (
                        f"White agent unresponsive after {consecutive_failures} "
                        f"consecutive failures."
                    )
                    break

                reasoning = f"Error occurred during communication (failure #{consecutive_failures})"
                # Fallback to first valid action or "look"
                action = current_admissible[0] if current_admissible else "look"
            else:
                # Reset failure counter on successful communication
                if consecutive_failures > 0:
                    LOGGER.info(
                        f"White agent communication recovered after {consecutive_failures} failures"
                    )
                consecutive_failures = 0

            # Execute action
            step_start_time = time.time()
            action_valid = True
            action_error = None

            try:
                step_result = self._env.step(action)
                # Update admissible commands for next step
                current_admissible = step_result.admissible_commands or []
            except TextWorldEnvironmentError as exc:
                LOGGER.error(f"Environment error: {exc}", exc_info=True)
                action_valid = False
                action_error = str(exc)
                terminated_early = True
                termination_reason = f"Environment error: {exc}"
                break

            step_duration = time.time() - step_start_time

            # Detect observation changes
            observation_changed = current_observation != step_result.observation

            # Assess reasoning quality indicators
            reasoning_length = len(reasoning) if reasoning else 0
            reasoning_coherent = bool(reasoning and len(reasoning) > 10)

            # Store trajectory step
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

            # Callback for progress reporting
            if on_step_complete:
                on_step_complete(step + 1, action, step_result.reward, step_result.done)

            # Consolidated step completion log
            LOGGER.info(
                f"[Step {step + 1}/{max_steps}] Result: "
                f"reward={step_result.reward:.1f}, done={step_result.done}, valid={action_valid}"
            )

            if step_result.done:
                LOGGER.info(f"Episode complete (done=True) after {step + 1} steps")
                break

        metrics = self._env.metrics()
        return EpisodeResult(
            trajectory=trajectory,
            goal=goal,
            success=bool(metrics.get("success", False)),
            step_count=len(trajectory),
            expert_plan=self._env.walkthrough(),
            terminated_early=terminated_early,
            termination_reason=termination_reason,
        )

    def _parse_command(self, white_response) -> Tuple[str, str]:
        """Extract reasoning and command from white agent's response.

        Args:
            white_response: Response message from white agent

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

    def _validate_and_correct_action(
        self,
        action: str,
        valid_actions: list[str],
    ) -> Tuple[str, bool]:
        """Validate action against valid_actions list.

        Args:
            action: The action string from the white agent
            valid_actions: List of admissible commands from the simulator

        Returns:
            Tuple of (corrected_action, was_valid)
        """
        if not valid_actions:
            return action, True  # No validation possible

        # Exact match
        if action in valid_actions:
            return action, True

        # Case-insensitive match
        action_lower = action.lower().strip()
        for valid in valid_actions:
            if valid.lower().strip() == action_lower:
                return valid, True  # Return the correctly-cased version

        # Fuzzy match: find closest valid action using substring matching
        for valid in valid_actions:
            valid_lower = valid.lower()
            # Check if action is a substring or vice versa
            if action_lower in valid_lower or valid_lower in action_lower:
                LOGGER.warning(
                    f"Corrected invalid action '{action}' to '{valid}' (substring match)"
                )
                return valid, False

        # Try prefix matching for common action patterns
        action_parts = action_lower.split()
        if action_parts:
            action_prefix = action_parts[0]  # e.g., "go", "take", "open"
            for valid in valid_actions:
                if valid.lower().startswith(action_prefix):
                    # Found an action with the same verb - use it
                    LOGGER.warning(
                        f"Corrected invalid action '{action}' to '{valid}' (prefix match)"
                    )
                    return valid, False

        # Default to first valid action or "look"
        fallback = valid_actions[0] if valid_actions else "look"
        LOGGER.warning(
            f"Invalid action '{action}' not found in valid_actions, "
            f"falling back to '{fallback}'"
        )
        return fallback, False

    @staticmethod
    def _format_initial_message(
        goal: str,
        observation: str,
        admissible_commands: list[str],
        *,
        ground_truth_actions: Optional[List[str]] = None,
    ) -> str:
        """Format first message to white agent with goal and valid actions.
        
        Args:
            goal: Task goal description
            observation: Initial observation
            admissible_commands: List of valid actions
            ground_truth_actions: Optional ground truth actions for oracle profile
        """
        # Format admissible commands as a numbered list for clarity
        valid_actions_str = (
            "\n".join(f"- {cmd}" for cmd in admissible_commands)
            if admissible_commands
            else "- look\n- inventory"
        )

        # Include ground truth actions if provided (for oracle profile)
        ground_truth_section = ""
        if ground_truth_actions:
            gt_actions_str = ",".join(ground_truth_actions)
            ground_truth_section = f"""
<ground_truth_actions>
{gt_actions_str}
</ground_truth_actions>
"""

        return f"""You are playing a TextWorld household task game.

<goal>
{goal}
</goal>

<goal_hint>
Note: Goals specify receptacle TYPES, not specific instances. "put X on cabinet" means ANY cabinet (1, 2, 3, etc.) satisfies the goal. "put some saltshaker" means ANY saltshaker instance works.
For pick-and-place: FIRST find and take the object, THEN place it on ANY matching receptacle.
</goal_hint>

<observation>
{observation}
</observation>

<valid_actions>
You MUST choose exactly one action from this list:
{valid_actions_str}
</valid_actions>
{ground_truth_section}
IMPORTANT: You can ONLY execute actions from the <valid_actions> list above. Any other action will fail.

Please respond with your reasoning in <reasoning>...</reasoning> tags, then your chosen action in <command>...</command> tags.
The command must be copied exactly from the valid_actions list.
"""

    @staticmethod
    def _format_observation_message(
        goal: str, observation: str, admissible_commands: list[str]
    ) -> str:
        """Format subsequent observation messages with goal reminder and valid actions."""
        # Format admissible commands as a list
        valid_actions_str = (
            "\n".join(f"- {cmd}" for cmd in admissible_commands)
            if admissible_commands
            else "- look\n- inventory"
        )

        return f"""<goal>
{goal}
</goal>

<observation>
{observation}
</observation>

<valid_actions>
You MUST choose exactly one action from this list:
{valid_actions_str}
</valid_actions>

What is your next command? Use <reasoning>...</reasoning> then <command>...</command> tags.
The command must be copied exactly from the valid_actions list.
"""


__all__ = ["EpisodeRunner", "EpisodeResult", "TrajectoryStep"]
