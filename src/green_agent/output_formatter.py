"""Output formatting for episode execution."""

from __future__ import annotations

from typing import List, Optional, Protocol

from .green_assessor import TrajectoryEval


class TerminalColors:
    """ANSI terminal color codes - consolidated in one place."""

    # Text styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    RESET = "\033[0m"

    # Agent colors
    GREEN_FG = "\033[38;5;46m"  # GREEN agent (environment/orchestrator)
    WHITE_FG = "\033[38;5;15m"  # WHITE agent (agent under test)
    YELLOW_FG = "\033[38;5;220m"  # JUDGE (LLM evaluator)
    CYAN_FG = "\033[38;5;51m"  # Baseline trajectory

    # Status colors
    RED_FG = "\033[38;5;196m"  # Failure/errors
    ORANGE_FG = "\033[38;5;208m"  # Warnings/medium

    # Dividers
    DIV = "─" * 80  # Single line divider
    DIV_DOUBLE = "═" * 80  # Double line for major sections
    DIV_LIGHT = "┄" * 80  # Light divider for subsections


def _truncate(text: str, max_len: int = 100) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _format_numbered_list(items: List[str], indent: int = 4) -> List[str]:
    """Format items as a numbered list with proper indentation."""
    lines = []
    prefix = " " * indent
    for i, item in enumerate(items, 1):
        # Truncate long items
        item_text = _truncate(item, 60)
        lines.append(f"{prefix}{i}. {item_text}")
    return lines


def _format_numbered_list_limited(
    items: List[str], *, max_items: int = 10, indent: int = 4
) -> List[str]:
    """Format items as a numbered list, but limit length to keep demo output readable."""
    shown = items[:max_items]
    lines = _format_numbered_list(shown, indent=indent)
    remaining = len(items) - len(shown)
    if remaining > 0:
        prefix = " " * indent
        lines.append(f"{prefix}… ({remaining} more)")
    return lines


def _score_color(score: float) -> str:
    """Get color based on score value."""
    if score >= 7.0:
        return TerminalColors.GREEN_FG
    elif score >= 4.0:
        return TerminalColors.YELLOW_FG
    else:
        return TerminalColors.RED_FG


class OutputFormatter(Protocol):
    """Protocol for episode output formatting."""

    def format_task_intro(self, goal: str, observation: str) -> str:
        """Format task introduction."""
        ...

    def format_step_start(self, step: int, max_steps: int, observation: str) -> str:
        """Format the beginning of a step (green agent's turn)."""
        ...

    def format_white_response(self, reasoning: str, action: str) -> str:
        """Format white agent's response."""
        ...

    def format_step_result(
        self,
        step: int,
        action: str,
        reward: float,
        done: bool,
        success: bool,
        goal: str,
    ) -> str:
        """Format step completion result."""
        ...

    def format_evaluation(self, eval_result: TrajectoryEval, goal: str) -> str:
        """Format final evaluation report."""
        ...

    def format_error(self, message: str) -> str:
        """Format error message."""
        ...


class DemoFormatter:
    """Human-friendly colored terminal output for demo mode."""

    def __init__(self) -> None:
        """Initialize demo formatter with state tracking."""
        self._initial_observation: str | None = None

    def format_task_intro(self, goal: str, observation: str) -> str:
        """Format task introduction with colored output.

        Note: Does NOT print the full observation here to avoid duplication
        with Step 1's GREEN observation. Just shows the goal and hints.
        """
        # Store initial observation to detect if step 1 should skip it
        self._initial_observation = observation

        actions_hint = "go to <recep>, open/close <recep>, take/move <obj>, inventory, look, examine"
        c = TerminalColors
        lines = [
            f"{c.DIV}",
            f"{c.BOLD}— Task Introduction —{c.RESET}",
            "",
            f"Goal: {c.BOLD}{goal}{c.RESET}",
            f"Actions: {c.DIM}{actions_hint}{c.RESET}",
        ]
        return "\n".join(lines)

    def format_step_start(self, step: int, max_steps: int, observation: str) -> str:
        """Format the beginning of a step with colored output."""
        c = TerminalColors
        lines = [
            c.DIV,
            "",
            f"{c.BOLD}Step {step}{c.RESET}",
            "",
            f"{c.BOLD}{c.GREEN_FG}GREEN{c.RESET}:",
        ]

        # For step 1, show full observation (since task intro no longer shows it)
        # For subsequent steps, show observation (environment feedback)
        lines.append(f"Observation: {c.BOLD}{observation}{c.RESET}")
        lines.append("")

        return "\n".join(lines)

    def format_white_response(self, reasoning: str, action: str) -> str:
        """Format white agent's response with colored output."""
        c = TerminalColors
        lines = [f"{c.BOLD}{c.WHITE_FG}WHITE{c.RESET}:"]
        if reasoning:
            reasoning_snippet = _truncate(reasoning.replace("\n", " "), 200)
            lines.append(f"Reasoning: {reasoning_snippet}")
        lines.append(f"Command: {c.BOLD}{action}{c.RESET}")
        return "\n".join(lines)

    def format_step_result(
        self,
        step: int,
        action: str,
        reward: float,
        done: bool,
        success: bool,
        goal: str,
    ) -> str:
        """Format step completion with task status (only show on completion)."""
        if done:
            c = TerminalColors
            if success:
                status_str = f"{c.GREEN_FG}✓ SUCCESS{c.RESET}"
            else:
                status_str = f"{c.RED_FG}✗ FAILED{c.RESET}"
            return f"\nTask: {_truncate(goal, 80)}\nOutcome: {status_str}\n"
        return ""

    def format_evaluation(self, eval_result: TrajectoryEval, goal: str) -> str:
        """Format final evaluation with consistent GREEN/WHITE visual style."""
        c = TerminalColors
        lines = []

        # Major section header
        lines.append("")
        lines.append(c.DIV_DOUBLE)
        lines.append(f"{c.BOLD}EVALUATION{c.RESET}")
        lines.append(c.DIV_DOUBLE)
        lines.append("")

        # Task outcome summary
        if eval_result.success:
            outcome = f"{c.GREEN_FG}✓ SUCCESS{c.RESET}"
        else:
            outcome = f"{c.RED_FG}✗ FAILED{c.RESET}"

        lines.append(f"Task: {_truncate(goal, 70)}")
        lines.append(
            f"Outcome: {outcome} in {c.BOLD}{eval_result.steps}{c.RESET} steps"
        )
        lines.append("")

        # Scores section
        lines.append(c.DIV)
        lines.append(f"{c.BOLD}{c.YELLOW_FG}JUDGE{c.RESET}: Scoring")
        lines.append("")

        # Correctness
        lines.extend(
            self._format_score_block(
                "Correctness",
                eval_result.correctness,
                eval_result.notes.get("correctness_reason", ""),
                f"success={eval_result.success} → {eval_result.correctness:.1f}",
            )
        )

        # Efficiency
        efficiency_detail = self._format_efficiency_detail(eval_result)
        lines.extend(
            self._format_score_block(
                "Efficiency",
                eval_result.efficiency,
                efficiency_detail,
                None,  # Calculation shown in detail
            )
        )

        # Strategy
        strategy_assessment = eval_result.notes.get("strategy_rationale", "")
        if not strategy_assessment:
            strategy_assessment = eval_result.notes.get("strategy_band", "")
        llm_tag = (
            f" {c.DIM}[LLM]{c.RESET}"
            if eval_result.features.get("llm_evaluated")
            else ""
        )
        lines.extend(
            self._format_score_block(
                f"Strategy{llm_tag}", eval_result.strategy, strategy_assessment, None
            )
        )

        # Reasoning
        reasoning_assessment = eval_result.notes.get("reasoning_rationale", "")
        if not reasoning_assessment:
            reasoning_assessment = eval_result.notes.get("reasoning_band", "")
        lines.extend(
            self._format_score_block(
                f"Reasoning{llm_tag}", eval_result.reasoning, reasoning_assessment, None
            )
        )

        # Overall score
        lines.append(c.DIV)
        lines.append(f"{c.BOLD}OVERALL SCORE COMPUTATION{c.RESET}")
        lines.append("")

        # Show inputs to computation
        lines.append(f"  {c.DIM}Inputs:{c.RESET}")
        lines.append(f"    Correctness (heuristic): {eval_result.correctness:.1f}")
        lines.append(f"    Efficiency (heuristic):  {eval_result.efficiency:.1f}")
        llm_tag = (
            " (LLM)" if eval_result.features.get("llm_evaluated") else " (heuristic)"
        )
        lines.append(f"    Strategy{llm_tag}:    {eval_result.strategy:.1f}")
        lines.append(f"    Reasoning{llm_tag}:   {eval_result.reasoning:.1f}")
        lines.append("")

        overall_color = _score_color(eval_result.overall)
        lines.append(
            f"  {c.DIM}Weights: correctness×0.30 + efficiency×0.20 + strategy×0.25 + reasoning×0.25{c.RESET}"
        )
        lines.append(
            f"  {c.DIM}= {eval_result.correctness:.1f}×0.30 + {eval_result.efficiency:.1f}×0.20 + {eval_result.strategy:.1f}×0.25 + {eval_result.reasoning:.1f}×0.25{c.RESET}"
        )
        lines.append("")
        lines.append(
            f"  {c.BOLD}Final Score:{c.RESET} {overall_color}{c.BOLD}{eval_result.overall:.1f}{c.RESET} / 10"
        )
        lines.append("")

        # Diagnostics (informational; not included in final score)
        # Note: We intentionally do NOT print baseline-vs-agent trajectory comparison here,
        # because it is not part of the quantitative evaluation.
        expert_actions = eval_result.features.get("expert_actions")
        if expert_actions:
            lines.append(c.DIV)
            lines.append(
                f"{c.BOLD}{c.DIM}DIAGNOSTICS{c.RESET}: Informational (not included in score)"
            )
            lines.append("")

            # Display baseline source from evaluation notes
            baseline_source = eval_result.notes.get("baseline_source", "handcoded_expert")
            baseline_label = {
                "ground_truth": "Ground Truth (optimal human demonstration)",
                "handcoded_expert": "ALFWorld handcoded expert",
            }.get(baseline_source, baseline_source)
            lines.append(
                f"{c.BOLD}{c.CYAN_FG}BASELINE TRAJECTORY{c.RESET}: {baseline_label}"
            )
            lines.append("")
            expert_steps = eval_result.features.get("expert_steps", len(expert_actions))
            lines.append(f"  Steps: {c.BOLD}{expert_steps}{c.RESET}")
            lines.append("  Actions:")
            lines.extend(_format_numbered_list_limited(expert_actions, max_items=10))
            lines.append("")

        # Failure patterns (if any)
        if eval_result.failure_patterns:
            lines.append(c.DIV)
            lines.append(
                f"{c.BOLD}{c.RED_FG}ISSUES{c.RESET}: Detected Failure Patterns"
            )
            lines.append("")
            for fp in eval_result.failure_patterns:
                severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                    fp.severity, "•"
                )
                lines.append(
                    f"  {severity_icon} {c.BOLD}{fp.name}{c.RESET} ({fp.severity})"
                )
                lines.append(f"     {c.DIM}{_truncate(fp.description, 60)}{c.RESET}")
                if fp.evidence:
                    lines.append(
                        f"     Evidence: {c.DIM}{_truncate(fp.evidence[0], 50)}{c.RESET}"
                    )
            lines.append("")

        lines.append(c.DIV_DOUBLE)

        return "\n".join(lines)

    def _format_score_block(
        self, name: str, score: float, assessment: str, calculation: Optional[str]
    ) -> List[str]:
        """Format a single score section with consistent style."""
        c = TerminalColors
        color = _score_color(score)

        lines = [f"  {c.BOLD}{name}{c.RESET}: {color}{score:.1f}{c.RESET} / 10"]

        if calculation:
            lines.append(f"    {c.DIM}{calculation}{c.RESET}")

        if assessment:
            # Truncate and format assessment
            assessment_text = _truncate(assessment.replace("\n", " "), 80)
            lines.append(f"    {c.DIM}{assessment_text}{c.RESET}")

        lines.append("")
        return lines

    def _format_efficiency_detail(self, eval_result: TrajectoryEval) -> str:
        """Format efficiency calculation details."""
        expert_steps = eval_result.features.get("expert_steps")
        ratio = eval_result.notes.get("efficiency_ratio", 0)
        overhead = eval_result.features.get("step_overhead", 0)

        # Display baseline source from evaluation notes
        baseline_source = eval_result.notes.get("baseline_source", "handcoded_expert")
        baseline_label = {
            "ground_truth": "Ground Truth",
            "handcoded_expert": "Handcoded Expert",
        }.get(baseline_source, baseline_source)
        return f"Agent={eval_result.steps}, Baseline={expert_steps} ({baseline_label}), Ratio={ratio:.2f}x, Overhead={overhead}"

    def format_error(self, message: str) -> str:
        """Format error message for demo mode."""
        c = TerminalColors
        return f"{c.RED_FG}⚠ Error:{c.RESET} {message}"


class LoggingFormatter:
    """Structured logging output (default non-demo mode)."""

    def format_task_intro(self, goal: str, observation: str) -> str:
        """Return empty - logging is handled via LOGGER in the caller."""
        return ""

    def format_step_start(self, step: int, max_steps: int, observation: str) -> str:
        """Return empty - logging is handled via LOGGER in the caller."""
        return ""

    def format_white_response(self, reasoning: str, action: str) -> str:
        """Return empty - logging is handled via LOGGER in the caller."""
        return ""

    def format_step_result(
        self,
        step: int,
        action: str,
        reward: float,
        done: bool,
        success: bool,
        goal: str,
    ) -> str:
        """Return empty - logging is handled via LOGGER in the caller."""
        return ""

    def format_evaluation(self, eval_result: TrajectoryEval, goal: str) -> str:
        """Return empty - logging is handled via LOGGER in the caller."""
        return ""

    def format_error(self, message: str) -> str:
        """Return empty - logging is handled via LOGGER in the caller."""
        return ""


class CollectingFormatter:
    """Formatter that collects output for A2A response messages."""

    def format_task_intro(self, goal: str, observation: str) -> str:
        """Format task introduction for message collection."""
        actions_hint = "- go to <recep>, open/close <recep>, take/move <obj>, inventory, look, examine"
        return (
            "Task Introduction\n"
            f"- What is the task? {goal}\n"
            f"- What does the environment look like? {_truncate(observation, 300)}\n"
            f"- What actions can each agent take? {actions_hint}"
        )

    def format_step_start(self, step: int, max_steps: int, observation: str) -> str:
        """Return empty - step progress reported via format_step_result."""
        return ""

    def format_white_response(self, reasoning: str, action: str) -> str:
        """Return empty - white response not included in collected output."""
        return ""

    def format_step_result(
        self,
        step: int,
        action: str,
        reward: float,
        done: bool,
        success: bool,
        goal: str,
    ) -> str:
        """Format step completion for message collection."""
        return f"Step {step}: '{action}' → reward={reward:.2f}, done={done}"

    def format_evaluation(self, eval_result: TrajectoryEval, goal: str) -> str:
        """Format final evaluation for message collection."""
        import json
        from .green_assessor import print_task_eval, STEP_BUDGET

        report = print_task_eval(eval_result, task_text=goal, step_budget=STEP_BUDGET)
        eval_json_payload = json.dumps(eval_result.to_dict())
        eval_tag = f"<eval_json>{eval_json_payload}</eval_json>"
        return f"{report}\n{eval_tag}"

    def format_error(self, message: str) -> str:
        """Format error message for message collection."""
        return f"Error: {message}"


def get_formatter(demo_mode: bool = False) -> OutputFormatter:
    """Get appropriate formatter based on mode.

    Args:
        demo_mode: If True, return DemoFormatter for human-readable output.
                   If False, return LoggingFormatter (LOGGER handles output).

    Returns:
        OutputFormatter instance
    """
    if demo_mode:
        return DemoFormatter()
    return LoggingFormatter()


__all__ = [
    "TerminalColors",
    "OutputFormatter",
    "DemoFormatter",
    "LoggingFormatter",
    "CollectingFormatter",
    "get_formatter",
]
