"""LLM-as-a-judge evaluator implementation."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..utils.vllm_client import completion as vllm_completion

LOGGER = logging.getLogger(__name__)


def _format_trajectory_for_llm_eval(trajectory: List[Dict[str, Any]]) -> str:
    """Format trajectory with reasoning for LLM evaluation prompt."""
    lines = []
    for step_data in trajectory:
        step_num = step_data.get("step", "?")
        reasoning = step_data.get("reasoning", "")
        action = step_data.get("action", "")
        observation = step_data.get("observation", "")

        # Format reasoning (truncate if very long)
        if reasoning:
            reasoning_display = (
                reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
            )
            lines.append(f"Step {step_num}: [Reasoning: {reasoning_display}]")
        else:
            lines.append(f"Step {step_num}: [No reasoning provided]")

        lines.append(f"  Action: {action}")

        # Include observation snippet for context
        if observation:
            obs_snippet = observation[:120].replace("\n", " ")
            lines.append(
                f"  Result: {obs_snippet}{'...' if len(observation) > 120 else ''}"
            )
        lines.append("")

    return "\n".join(lines)


async def evaluate_with_llm(
    goal: str,
    trajectory: List[Dict[str, Any]],
    success: bool,
    step_budget: int = 50,
) -> Optional[Dict[str, Any]]:
    """
    Single LLM call to evaluate reasoning and strategy quality.

    Args:
        goal: Task goal description
        trajectory: List of trajectory step dictionaries
        success: Whether task was completed successfully
        step_budget: Maximum allowed steps

    Returns:
        Dict with keys: reasoning_score, strategy_score,
        reasoning_rationale, strategy_rationale, notable_moments

        Returns None on failure (caller should fall back to heuristics).
    """
    steps_used = len(trajectory)
    outcome_str = "Success" if success else "Failure"
    trajectory_text = _format_trajectory_for_llm_eval(trajectory)

    prompt = f"""You are evaluating an AI agent's performance on a TextWorld household task.

TASK GOAL: {goal}
OUTCOME: {outcome_str} in {steps_used}/{step_budget} steps

TRAJECTORY:
{trajectory_text}

Evaluate TWO dimensions on a 1-10 scale:

REASONING QUALITY: Does the agent's reasoning...
- Explicitly reference the goal object and target location?
- Ground explanations in what it actually observes (objects, locations)?
- Explain WHY it chose each action (not just what it does)?
- Adapt when objects aren't where expected or actions fail?
- Avoid copy-paste or generic "I will explore" statements?

STRATEGY QUALITY: Does the agent's approach...
- Check likely locations first (countertops/tables for visible items, cabinets/drawers for stored)?
- Follow correct task order: find → pick up → transform if needed → navigate → place?
- Minimize backtracking and revisits to same locations?
- Open containers systematically when searching?
- Recover quickly from "nothing happens" or error observations?

TextWorld-specific score anchors:
- 9-10: Checks 2-3 high-probability locations, correct subgoal order, adaptive reasoning
- 7-8: Mostly efficient search, occasional unnecessary detours, reasoning references goal
- 5-6: Systematic but poorly prioritized (e.g., checks every cabinet), shallow reasoning
- 3-4: Frequent revisits, generic reasoning like "let me explore", poor adaptation
- 1-2: Random wandering, no goal awareness, repeated identical actions (stuck loop)

Respond ONLY with valid JSON (no markdown, no explanation):
{{"reasoning_score": <float 1-10>, "reasoning_rationale": "<cite 1 specific step>", "strategy_score": <float 1-10>, "strategy_rationale": "<cite 1 specific step>", "notable_moments": ["step N: observation"]}}"""

    try:
        LOGGER.debug(
            f"[LLM Judge] Evaluating agent trajectory: goal='{goal[:50]}...', "
            f"steps={steps_used}, outcome={outcome_str}"
        )

        response = vllm_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        content = response.content.strip()

        # Log raw LLM response before parsing
        LOGGER.debug(f"[LLM Judge] Raw response:\n{content[:500]}{'...' if len(content) > 500 else ''}")

        # Try to parse JSON response
        # Handle potential markdown code blocks
        if content.startswith("```"):
            # Extract JSON from code block
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
            if json_match:
                content = json_match.group(1).strip()

        result = json.loads(content)

        # Validate required fields
        required_fields = [
            "reasoning_score",
            "strategy_score",
            "reasoning_rationale",
            "strategy_rationale",
        ]
        for field in required_fields:
            if field not in result:
                LOGGER.warning(f"LLM evaluation missing required field: {field}")
                return None

        # Log raw scores before clamping
        LOGGER.debug(
            f"[LLM Judge] Parsed scores (pre-clamp): "
            f"reasoning={result['reasoning_score']}, strategy={result['strategy_score']}"
        )

        # Clamp scores to valid range
        result["reasoning_score"] = max(
            1.0, min(10.0, float(result["reasoning_score"]))
        )
        result["strategy_score"] = max(1.0, min(10.0, float(result["strategy_score"])))

        # Ensure notable_moments is a list
        if "notable_moments" not in result:
            result["notable_moments"] = []
        elif not isinstance(result["notable_moments"], list):
            result["notable_moments"] = [str(result["notable_moments"])]

        LOGGER.info(
            f"[LLM Judge] Final scores: reasoning={result['reasoning_score']:.1f}, "
            f"strategy={result['strategy_score']:.1f}"
        )

        return result

    except json.JSONDecodeError as e:
        LOGGER.warning(f"LLM evaluation JSON parse failed: {e}")
        return None
    except Exception as exc:
        LOGGER.warning(f"LLM evaluation failed: {exc}")
        return None


def _format_expert_trajectory(expert_trajectory: Any) -> str:
    """Format expert trajectory for LLM comparison."""
    lines = [f"Initial observation: {expert_trajectory.initial_observation[:200]}...\n"]

    for step in expert_trajectory.steps:
        lines.append(f"Step {step.step_num}: Action: {step.action}")
        obs_snippet = step.observation[:100].replace("\n", " ")
        lines.append(
            f"  Result: {obs_snippet}{'...' if len(step.observation) > 100 else ''}"
        )

    lines.append(f"\nOutcome: {'Success' if expert_trajectory.success else 'Failure'}")
    lines.append(f"Total steps: {expert_trajectory.step_count}")

    return "\n".join(lines)


async def compare_trajectories_with_llm(
    goal: str,
    agent_trajectory: List[Dict[str, Any]],
    expert_trajectory: Any,  # ExpertTrajectory type
    agent_success: bool,
) -> Optional[Dict[str, Any]]:
    """
    Use LLM-as-judge to compare agent trajectory against expert trajectory.

    Args:
        goal: Task goal description
        agent_trajectory: Agent's trajectory steps
        expert_trajectory: ExpertTrajectory or baseline trajectory for comparison
        agent_success: Whether agent completed the task

    Returns:
        Dict with comparison scores and analysis, or None on failure
    """
    agent_steps = len(agent_trajectory)
    expert_steps = expert_trajectory.step_count
    agent_outcome = "Success" if agent_success else "Failure"
    # Note: expert_trajectory.success is used via expert_trajectory_text

    agent_trajectory_text = _format_trajectory_for_llm_eval(agent_trajectory)
    expert_trajectory_text = _format_expert_trajectory(expert_trajectory)

    # Log expert trajectory immediately before LLM comparison
    LOGGER.debug(
        f"[Expert Trajectory] goal='{goal[:50]}...', expert_steps={expert_steps}, "
        f"expert_success={expert_trajectory.success}"
    )
    LOGGER.debug(f"[Expert Trajectory] Actions: {expert_trajectory.actions}")
    LOGGER.debug(
        f"[Expert Trajectory] Full formatted:\n{expert_trajectory_text[:800]}"
        f"{'...' if len(expert_trajectory_text) > 800 else ''}"
    )

    prompt = f"""You are comparing an AI agent's trajectory against a baseline trajectory on the same TextWorld task.

Note: The baseline trajectory represents optimal or near-optimal performance (either human demonstration or domain-aware heuristics).

TASK GOAL: {goal}

=== BASELINE TRAJECTORY ===
{expert_trajectory_text}

=== AGENT TRAJECTORY (Being Evaluated) ===
Outcome: {agent_outcome} in {agent_steps} steps (baseline used {expert_steps} steps)

{agent_trajectory_text}

Evaluate the agent's performance by comparing to the expert:

1. PATH EQUIVALENCE (1-10): Did the agent achieve the same milestones/subgoals as the expert?
   - 10: Identical or functionally equivalent path
   - 7-9: Same key actions, minor detours
   - 4-6: Reached goal but very different approach
   - 1-3: Failed or wildly different path

2. EFFICIENCY COMPARISON (1-10): How does agent efficiency compare?
   - 10: Same or fewer steps than expert
   - 7-9: 1.5x expert steps or less
   - 4-6: 2-3x expert steps
   - 1-3: >3x expert steps or failed

3. MILESTONE COVERAGE: What percentage of expert's key actions did the agent perform?
   Key actions include: picking up target objects, going to target locations, placing objects correctly.

4. DEVIATION ANALYSIS: Where did the agent deviate from the baseline path?

Respond ONLY with valid JSON (no markdown):
{{"path_equivalence": <float 1-10>, "efficiency_comparison": <float 1-10>, "milestone_coverage": <float 0-1>, "deviation_analysis": "<brief explanation>", "agent_strengths": ["strength1"], "agent_weaknesses": ["weakness1"]}}"""

    try:
        LOGGER.debug(
            f"[Expert Comparison] Calling LLM to compare trajectories: "
            f"agent_steps={agent_steps}, expert_steps={expert_steps}"
        )

        response = vllm_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        content = response.content.strip()

        # Log raw LLM response before parsing
        LOGGER.debug(
            f"[Expert Comparison] Raw LLM response:\n{content[:600]}"
            f"{'...' if len(content) > 600 else ''}"
        )

        # Handle markdown code blocks
        if content.startswith("```"):
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
            if json_match:
                content = json_match.group(1).strip()

        result = json.loads(content)

        # Log parsed scores before clamping
        LOGGER.debug(
            f"[Expert Comparison] Parsed (pre-clamp): "
            f"path_equiv={result.get('path_equivalence')}, "
            f"efficiency={result.get('efficiency_comparison')}, "
            f"milestone_cov={result.get('milestone_coverage')}"
        )

        # Validate and clamp scores
        result["path_equivalence"] = max(
            1.0, min(10.0, float(result.get("path_equivalence", 5.0)))
        )
        result["efficiency_comparison"] = max(
            1.0, min(10.0, float(result.get("efficiency_comparison", 5.0)))
        )
        result["milestone_coverage"] = max(
            0.0, min(1.0, float(result.get("milestone_coverage", 0.5)))
        )

        # Ensure lists exist
        if "agent_strengths" not in result:
            result["agent_strengths"] = []
        if "agent_weaknesses" not in result:
            result["agent_weaknesses"] = []
        if "deviation_analysis" not in result:
            result["deviation_analysis"] = "No analysis provided"

        # Add quantitative metrics
        result["expert_steps"] = expert_steps
        result["agent_steps"] = agent_steps
        result["step_ratio"] = (
            round(agent_steps / expert_steps, 2) if expert_steps > 0 else None
        )
        result["step_overhead"] = agent_steps - expert_steps

        LOGGER.info(
            f"[Expert Comparison] Final: path_equiv={result['path_equivalence']:.1f}, "
            f"efficiency={result['efficiency_comparison']:.1f}, "
            f"milestone_cov={result['milestone_coverage']:.0%}, "
            f"step_ratio={result['step_ratio']}x"
        )

        return result

    except json.JSONDecodeError as e:
        LOGGER.warning(f"Trajectory comparison JSON parse failed: {e}")
        return None
    except Exception as exc:
        LOGGER.warning(f"Trajectory comparison failed: {exc}")
        return None


__all__ = ["evaluate_with_llm", "compare_trajectories_with_llm"]
