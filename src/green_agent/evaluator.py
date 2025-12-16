"""LLM-as-a-judge evaluator implementation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..utils.vllm_client import completion as vllm_completion

from .rubric import EvaluationRubric, ModelSettings

LOGGER = logging.getLogger(__name__)


@dataclass
class CategoryRating:
    """Rating for a single evaluation category."""

    score: float
    weight: float
    criterion_name: str

    def weighted_score(self) -> float:
        """Calculate weighted score."""
        return self.score * self.weight


@dataclass
class ScoreBreakdown:
    """Structured breakdown of evaluation scores."""

    overall_rating: float
    category_ratings: Dict[str, CategoryRating] = field(default_factory=dict)
    qualitative_assessments: Dict[str, List[str]] = field(default_factory=dict)
    reasoning_trace_analysis: Dict[str, str] = field(default_factory=dict)
    quick_rating: Optional[float] = None
    detailed_reasoning: Optional[str] = None

    def compute_weighted_overall(self) -> float:
        """Compute weighted overall score from category ratings."""
        if not self.category_ratings:
            return self.overall_rating
        return sum(cr.weighted_score() for cr in self.category_ratings.values())


@dataclass
class ReasoningTraceAssessment:
    """Assessment of reasoning traces."""

    reasoning_quality: str
    planning_evidence: str
    error_handling: str


class LLMJudgeEvaluator:
    """LLM-as-a-judge evaluator using configurable rubric."""

    def __init__(self, rubric: EvaluationRubric):
        """Initialize evaluator with rubric.

        Args:
            rubric: Evaluation rubric configuration
        """
        self.rubric = rubric

    async def rate_trajectory_quick(
        self, goal: str, trajectory: List[Dict[str, Any]], success: bool
    ) -> float:
        """Quick numeric rating without detailed reasoning.

        Args:
            goal: Task goal description
            trajectory: List of trajectory step dictionaries
            success: Whether task was completed successfully

        Returns:
            Rating from 1-10
        """
        trajectory_text = self._format_trajectory(trajectory)

        prompt = f"""Rate this TextWorld agent's overall performance from {self.rubric.scoring_scale.min}-{self.rubric.scoring_scale.max}.

Goal: {goal}
Outcome: {"Success" if success else "Failure"}
Steps: {len(trajectory)}/{50}

Trajectory:
{trajectory_text}

Quick scoring guide:
- 9-10: Success + efficient (<15 steps) + clear reasoning
- 7-8: Success + moderate steps (15-30) OR efficient failure with good strategy
- 5-6: Success but inefficient (30-50 steps) OR close failure with decent approach
- 3-4: Failure with some goal-directed behavior
- 1-2: Failure with random/stuck behavior

Respond with ONLY a number (e.g., 7.5). No explanation."""

        model_settings = self.rubric.model_settings.get("quick_rating")
        if model_settings is None:
            model_settings = ModelSettings(
                model="Qwen/Qwen2.5-7B-Instruct", temperature=0.0
            )

        try:
            response = vllm_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=model_settings.temperature,
            )

            rating_text = response.content.strip()
            numbers = re.findall(r"\d+\.?\d*", rating_text)
            if numbers:
                rating = float(numbers[0])
                return max(
                    self.rubric.scoring_scale.min,
                    min(self.rubric.scoring_scale.max, rating),
                )
        except Exception as exc:
            LOGGER.error(f"Quick rating failed: {exc}")

        # Fallback heuristic
        if success:
            efficiency = max(0, 1.0 - len(trajectory) / 50.0)
            return self.rubric.scoring_scale.min + (
                (self.rubric.scoring_scale.max - self.rubric.scoring_scale.min)
                * (0.7 + 0.3 * efficiency)
            )
        return self.rubric.scoring_scale.min + (
            (self.rubric.scoring_scale.max - self.rubric.scoring_scale.min) * 0.3
        )

    async def rate_trajectory_detailed(
        self, goal: str, trajectory: List[Dict[str, Any]], success: bool
    ) -> ScoreBreakdown:
        """Detailed rating with structured breakdown using rubric.

        Args:
            goal: Task goal description
            trajectory: List of trajectory step dictionaries
            success: Whether task was completed successfully

        Returns:
            ScoreBreakdown with category ratings and qualitative assessments
        """
        trajectory_text = self._format_trajectory(trajectory)

        # Build prompt using rubric criteria
        prompt_parts = [
            "Evaluate this TextWorld agent's performance using the following criteria:",
            "",
        ]

        # Add criteria descriptions
        for key, criterion in self.rubric.criteria.items():
            prompt_parts.append(
                f"{criterion.name} (Weight: {criterion.weight:.0%}): {criterion.description}"
            )
            prompt_parts.append(f"  {criterion.prompt_template}")
            prompt_parts.append("")

        prompt_parts.extend(
            [
                f"Goal: {goal}",
                f"Success: {success}",
                "",
                "Trajectory:",
                trajectory_text,
                "",
            ]
        )

        # Request structured output
        prompt_parts.append("Provide your evaluation in the following format:")
        prompt_parts.append("")
        prompt_parts.append("Overall Rating: [number]")
        prompt_parts.append("")

        # Category ratings
        for key, criterion in self.rubric.criteria.items():
            prompt_parts.append(f"{criterion.name} Rating: [number]")
        prompt_parts.append("")

        # Qualitative assessments if enabled
        if self.rubric.qualitative_assessments.get("enabled", False):
            fields = self.rubric.qualitative_assessments.get("fields", [])
            if "strengths" in fields:
                prompt_parts.append("Strengths:")
                prompt_parts.append("- [strength 1]")
                prompt_parts.append("- [strength 2]")
                prompt_parts.append("")
            if "weaknesses" in fields:
                prompt_parts.append("Weaknesses:")
                prompt_parts.append("- [weakness 1]")
                prompt_parts.append("- [weakness 2]")
                prompt_parts.append("")
            if "notable_behaviors" in fields:
                prompt_parts.append("Notable Behaviors:")
                prompt_parts.append("- [behavior 1]")
                prompt_parts.append("")
            if "recommendations" in fields:
                prompt_parts.append("Recommendations:")
                prompt_parts.append("- [recommendation 1]")
                prompt_parts.append("")

        prompt_parts.append("Detailed Reasoning: [detailed analysis]")

        prompt = "\n".join(prompt_parts)

        model_settings = self.rubric.model_settings.get("detailed_rating")
        if model_settings is None:
            model_settings = ModelSettings(
                model="Qwen/Qwen2.5-7B-Instruct", temperature=0.3
            )

        try:
            response = vllm_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=model_settings.temperature,
            )

            content = response.content

            # Parse overall rating
            overall_rating = self._parse_overall_rating(content)

            # Parse category ratings
            category_ratings = self._parse_category_ratings(content)

            # Parse qualitative assessments
            qualitative_assessments = self._parse_qualitative_assessments(content)

            # Parse detailed reasoning
            detailed_reasoning = self._parse_detailed_reasoning(content)

            # Compute weighted overall if we have category ratings
            if category_ratings:
                weighted_overall = sum(
                    cr.weighted_score() for cr in category_ratings.values()
                )
                # Use weighted overall if it's reasonable, otherwise use parsed overall
                if abs(weighted_overall - overall_rating) < 2.0:
                    overall_rating = weighted_overall
            else:
                weighted_overall = overall_rating

            return ScoreBreakdown(
                overall_rating=overall_rating,
                category_ratings=category_ratings,
                qualitative_assessments=qualitative_assessments,
                detailed_reasoning=detailed_reasoning,
            )

        except Exception as exc:
            LOGGER.error(f"Detailed rating failed: {exc}", exc_info=True)

        # Fallback
        if success:
            return ScoreBreakdown(
                overall_rating=7.0,
                detailed_reasoning="Task completed successfully.",
            )
        return ScoreBreakdown(
            overall_rating=3.0,
            detailed_reasoning="Task not completed.",
        )

    async def assess_reasoning_traces(
        self, goal: str, trajectory: List[Dict[str, Any]]
    ) -> ReasoningTraceAssessment:
        """Analyze reasoning traces step-by-step.

        Args:
            goal: Task goal description
            trajectory: List of trajectory step dictionaries with reasoning

        Returns:
            ReasoningTraceAssessment with analysis of reasoning quality
        """
        trajectory_text = self._format_trajectory_with_reasoning(trajectory)

        prompt = f"""Analyze the reasoning traces provided by the TextWorld agent at each step.

Goal: {goal}

Trajectory with Reasoning:
{trajectory_text}

Provide analysis in the following format:

Reasoning Quality: [Does the agent explain WHY actions are chosen? Does it reference what it actually sees in observations? Are explanations unique per step or copy-pasted? Quote 1 example.]

Planning Evidence: [Does the agent show subgoal awareness (find→pick→transform→place)? Does it prioritize likely object locations? Does it systematically search containers? Quote 1 example.]

Error Handling: [How does the agent respond to "nothing happens", closed containers, or objects not found? Does it adapt strategy or repeat failed actions? Quote 1 example.]

Be specific and cite step numbers."""

        model_settings = self.rubric.model_settings.get("detailed_rating")
        if model_settings is None:
            model_settings = ModelSettings(
                model="Qwen/Qwen2.5-7B-Instruct", temperature=0.3
            )

        try:
            response = vllm_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=model_settings.temperature,
            )

            content = response.content

            reasoning_quality = self._extract_field(content, "Reasoning Quality:")
            planning_evidence = self._extract_field(content, "Planning Evidence:")
            error_handling = self._extract_field(content, "Error Handling:")

            return ReasoningTraceAssessment(
                reasoning_quality=reasoning_quality or "No assessment provided",
                planning_evidence=planning_evidence or "No planning evidence found",
                error_handling=error_handling
                or "No error handling patterns identified",
            )

        except Exception as exc:
            LOGGER.error(f"Reasoning trace assessment failed: {exc}", exc_info=True)

        return ReasoningTraceAssessment(
            reasoning_quality="Assessment failed",
            planning_evidence="Assessment failed",
            error_handling="Assessment failed",
        )

    def _format_trajectory(self, trajectory: List[Dict[str, Any]]) -> str:
        """Format trajectory for LLM evaluation."""
        lines = []
        for step_data in trajectory:
            reasoning = step_data.get("reasoning", "")
            action = step_data.get("action", "")
            reward = step_data.get("reward", 0.0)
            observation = step_data.get("observation", "")

            if reasoning:
                lines.append(
                    f"Step {step_data.get('step', '?')}: [{reasoning}] → {action} "
                    f"(reward: {reward:.2f})"
                )
            else:
                lines.append(
                    f"Step {step_data.get('step', '?')}: {action} "
                    f"(reward: {reward:.2f})"
                )

            # Include observation snippet for context (first 100 chars)
            if observation:
                obs_snippet = observation[:100].replace("\n", " ")
                lines.append(
                    f"  Observation: {obs_snippet}{'...' if len(observation) > 100 else ''}"
                )

        return "\n".join(lines)

    def _format_trajectory_with_reasoning(
        self, trajectory: List[Dict[str, Any]]
    ) -> str:
        """Format trajectory emphasizing reasoning traces."""
        lines = []
        for step_data in trajectory:
            step_num = step_data.get("step", "?")
            reasoning = step_data.get("reasoning", "")
            action = step_data.get("action", "")
            reward = step_data.get("reward", 0.0)
            observation = step_data.get("observation", "")

            lines.append(f"Step {step_num}:")
            if reasoning:
                lines.append(f"  Reasoning: {reasoning}")
            else:
                lines.append("  Reasoning: [none provided]")
            lines.append(f"  Action: {action}")
            lines.append(f"  Reward: {reward:.2f}")
            if observation:
                obs_snippet = observation[:150].replace("\n", " ")
                lines.append(
                    f"  Observation: {obs_snippet}{'...' if len(observation) > 150 else ''}"
                )
            lines.append("")

        return "\n".join(lines)

    def _parse_overall_rating(self, content: str) -> float:
        """Parse overall rating from LLM response."""
        # Try various patterns
        patterns = [
            r"Overall Rating:\s*(\d+\.?\d*)",
            r"Rating:\s*(\d+\.?\d*)",
            r"Overall:\s*(\d+\.?\d*)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                rating = float(match.group(1))
                return max(
                    self.rubric.scoring_scale.min,
                    min(self.rubric.scoring_scale.max, rating),
                )

        # Fallback: find first number
        numbers = re.findall(r"\d+\.?\d*", content)
        if numbers:
            rating = float(numbers[0])
            return max(
                self.rubric.scoring_scale.min,
                min(self.rubric.scoring_scale.max, rating),
            )

        return (self.rubric.scoring_scale.min + self.rubric.scoring_scale.max) / 2.0

    def _parse_category_ratings(self, content: str) -> Dict[str, CategoryRating]:
        """Parse category ratings from LLM response."""
        category_ratings = {}

        for key, criterion in self.rubric.criteria.items():
            # Try various patterns for this criterion
            patterns = [
                rf"{re.escape(criterion.name)}\s*Rating:\s*(\d+\.?\d*)",
                rf"{re.escape(key)}\s*Rating:\s*(\d+\.?\d*)",
                rf"{re.escape(criterion.name)}:\s*(\d+\.?\d*)",
            ]

            rating = None
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    rating = float(match.group(1))
                    break

            if rating is not None:
                rating = max(
                    self.rubric.scoring_scale.min,
                    min(self.rubric.scoring_scale.max, rating),
                )
                category_ratings[key] = CategoryRating(
                    score=rating, weight=criterion.weight, criterion_name=criterion.name
                )

        return category_ratings

    def _parse_qualitative_assessments(self, content: str) -> Dict[str, List[str]]:
        """Parse qualitative assessments from LLM response."""
        assessments = {}

        fields = self.rubric.qualitative_assessments.get("fields", [])
        for field_name in fields:
            items = self._extract_list_items(content, field_name)
            if items:
                assessments[field_name] = items

        return assessments

    def _extract_list_items(self, content: str, field_name: str) -> List[str]:
        """Extract bullet list items for a field."""
        # Look for field header followed by bullet points
        pattern = rf"{re.escape(field_name)}:?\s*\n((?:[-*•]\s*.+\n?)+)"
        match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
        if match:
            list_text = match.group(1)
            # Extract items (lines starting with -, *, or •)
            items = re.findall(
                r"[-*•]\s*(.+?)(?=\n[-*•]|\n\n|\Z)", list_text, re.MULTILINE
            )
            return [item.strip() for item in items if item.strip()]

        # Try alternative pattern with numbered list or just lines after header
        pattern = rf"{re.escape(field_name)}:?\s*\n((?:\d+\.\s*.+\n?)+)"
        match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
        if match:
            list_text = match.group(1)
            items = re.findall(
                r"\d+\.\s*(.+?)(?=\n\d+\.|\n\n|\Z)", list_text, re.MULTILINE
            )
            return [item.strip() for item in items if item.strip()]

        return []

    def _extract_field(self, content: str, field_name: str) -> Optional[str]:
        """Extract a single text field value."""
        pattern = rf"{re.escape(field_name)}\s*(.+?)(?=\n[A-Z][^:]+:|$)"
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _parse_detailed_reasoning(self, content: str) -> Optional[str]:
        """Parse detailed reasoning from LLM response."""
        patterns = [
            r"Detailed Reasoning:\s*(.+)",
            r"Reasoning:\s*(.+)",
            r"Analysis:\s*(.+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()

        # If no structured field found, return everything after the last rating
        return None


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
    import json

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
        response = vllm_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        content = response.content.strip()

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
            f"LLM evaluation: reasoning={result['reasoning_score']:.1f}, "
            f"strategy={result['strategy_score']:.1f}"
        )

        return result

    except json.JSONDecodeError as e:
        LOGGER.warning(f"LLM evaluation JSON parse failed: {e}")
        return None
    except Exception as exc:
        LOGGER.warning(f"LLM evaluation failed: {exc}")
        return None


__all__ = [
    "CategoryRating",
    "ScoreBreakdown",
    "ReasoningTraceAssessment",
    "LLMJudgeEvaluator",
    "evaluate_with_llm",
]
