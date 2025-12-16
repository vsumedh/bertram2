"""Assessment utilities for green agent multi-task evaluation."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Sequence, Optional


STEP_BUDGET = 50

# Scoring weights (aligned with evaluation_rubric.json v1.1)
WEIGHT_TASK_COMPLETION = 0.30
WEIGHT_EFFICIENCY = 0.20
WEIGHT_STRATEGY = 0.25
WEIGHT_REASONING = 0.25

# Aliases for backward compatibility
WEIGHT_CORRECTNESS = WEIGHT_TASK_COMPLETION


@dataclass
class FailurePattern:
    """Detected failure pattern in trajectory."""

    name: str
    description: str
    severity: str  # "high", "medium", "low"
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryEval:
    task_id: str
    success: bool
    steps: int
    correctness: float
    efficiency: float
    strategy: float
    reasoning: float
    overall: float
    quick: float
    features: Dict[str, Any]
    notes: Dict[str, Any] | str
    failure_patterns: List[FailurePattern] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["failure_patterns"] = [fp.to_dict() for fp in self.failure_patterns]
        return result


def _safe_mean(values: Sequence[float]) -> float:
    clean = [v for v in values if v is not None]
    return float(statistics.mean(clean)) if clean else 0.0


def _clamp(val: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, val))


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def detect_failure_patterns(
    trajectory_steps: List[Dict[str, Any]],
    success: bool,
    task_text: str,
) -> List[FailurePattern]:
    """Detect common failure patterns in the trajectory.

    Patterns detected:
    - stuck_loop: Agent repeats the same action 3+ times consecutively
    - wrong_order: Agent tries to place object before picking it up
    - target_confusion: Agent places object in wrong location
    - object_confusion: Agent picks up wrong object type
    - incomplete_transform: Agent skips required transformation
    - excessive_exploration: Agent visits 10+ locations without finding target
    """
    patterns: List[FailurePattern] = []

    if not trajectory_steps:
        return patterns

    actions = [str(step.get("action", "")).strip().lower() for step in trajectory_steps]
    observations = [
        str(step.get("observation", "")).lower() for step in trajectory_steps
    ]

    # Detect stuck_loop: same action 3+ times consecutively
    consecutive_count = 1
    for i in range(1, len(actions)):
        if actions[i] == actions[i - 1]:
            consecutive_count += 1
            if consecutive_count >= 3:
                patterns.append(
                    FailurePattern(
                        name="stuck_loop",
                        description="Agent repeats the same action 3+ times consecutively",
                        severity="high",
                        evidence=[
                            f"Repeated '{actions[i]}' at steps {i - consecutive_count + 2}-{i + 1}"
                        ],
                    )
                )
                break
        else:
            consecutive_count = 1

    # Detect wrong_order: put without prior take
    holding_object = False
    for i, (action, obs) in enumerate(zip(actions, observations)):
        if action.startswith("take ") or action.startswith("pick up "):
            if "you take" in obs or "you pick up" in obs:
                holding_object = True
        if action.startswith("put "):
            if not holding_object:
                patterns.append(
                    FailurePattern(
                        name="wrong_order",
                        description="Agent tries to place object before picking it up",
                        severity="medium",
                        evidence=[
                            f"Step {i + 1}: attempted '{action}' without holding object"
                        ],
                    )
                )
                break
            if "you put" in obs:
                holding_object = False

    # Detect excessive_exploration: 10+ unique locations visited without task progress
    unique_locs_before_action = set()
    found_target = False
    for i, action in enumerate(actions):
        if action.startswith("go to "):
            loc = action.replace("go to ", "", 1).strip()
            unique_locs_before_action.add(loc)
        if action.startswith("take ") or action.startswith("pick up "):
            found_target = True
            break

    if len(unique_locs_before_action) >= 10 and not found_target and not success:
        patterns.append(
            FailurePattern(
                name="excessive_exploration",
                description="Agent visits 10+ locations without finding target object",
                severity="low",
                evidence=[
                    f"Visited {len(unique_locs_before_action)} locations before finding target"
                ],
            )
        )

    # Detect target_confusion: placed in wrong location (heuristic based on task text)
    task_lower = task_text.lower()
    target_locs = []
    for keyword in ["in ", "on ", "to "]:
        if keyword in task_lower:
            idx = task_lower.rfind(keyword)
            potential_loc = task_lower[idx:].split()[1:3]
            target_locs.extend(potential_loc)

    if target_locs and not success:
        for i, action in enumerate(actions):
            if action.startswith("put "):
                # Check if the location in the put action matches expected target
                put_parts = action.split(" in ") + action.split(" on ")
                if len(put_parts) > 1:
                    actual_loc = put_parts[-1].strip()
                    if not any(t in actual_loc for t in target_locs):
                        patterns.append(
                            FailurePattern(
                                name="target_confusion",
                                description="Agent places object in wrong location",
                                severity="high",
                                evidence=[
                                    f"Step {i + 1}: placed in '{actual_loc}', expected one of {target_locs}"
                                ],
                            )
                        )
                        break

    # Detect incomplete_transform: task requires clean/heat/cool but not performed
    transform_keywords = {
        "clean": ["clean", "wash", "rinse"],
        "heat": ["heat", "warm", "cook", "microwave"],
        "cool": ["cool", "chill", "refrigerate", "cold"],
    }

    for transform_type, keywords in transform_keywords.items():
        if any(kw in task_lower for kw in keywords):
            transform_performed = any(
                any(kw in action for kw in keywords) for action in actions
            )
            if not transform_performed and not success:
                patterns.append(
                    FailurePattern(
                        name="incomplete_transform",
                        description=f"Agent skips required transformation ({transform_type})",
                        severity="high",
                        evidence=[
                            f"Task mentions '{transform_type}' but no corresponding action found"
                        ],
                    )
                )

    return patterns


def compute_weighted_overall(
    correctness: float,
    efficiency: float,
    strategy: float,
    reasoning: float,
) -> float:
    """Compute overall score with balanced weights.

    Weights: correctness=30%, efficiency=20%, strategy=25%, reasoning=25%
    """
    return round(
        correctness * WEIGHT_CORRECTNESS
        + efficiency * WEIGHT_EFFICIENCY
        + strategy * WEIGHT_STRATEGY
        + reasoning * WEIGHT_REASONING,
        2,
    )


def evaluate_trajectory(
    *,
    task_id: str,
    task_text: str,
    step_budget: int,
    trajectory_steps: List[Dict[str, Any]],
    env_success: bool,
) -> TrajectoryEval:
    steps_used = len(trajectory_steps)
    success = bool(env_success)

    # Detect failure patterns for diagnostic purposes
    failure_patterns = detect_failure_patterns(trajectory_steps, success, task_text)

    # Task Completion (was: Correctness)
    correctness = 10.0 if success else 0.0
    correctness_reason = (
        "Env reports success (goal state reached)."
        if success
        else "Env reports failure."
    )

    # Efficiency
    if success:
        ratio = steps_used / float(step_budget) if step_budget > 0 else 1.0
        efficiency = _clamp(10.0 * (1.0 - ratio))
    else:
        ratio = steps_used / float(step_budget) if step_budget > 0 else 1.0
        efficiency = 0.0

    # Strategy features
    unique_locs = set()
    revisit_count = 0
    navigation_steps = 0
    no_effect = 0

    for step in trajectory_steps:
        action = str(step.get("action", "")).strip().lower()
        obs = str(step.get("observation", "")).lower()
        if action.startswith("go to "):
            navigation_steps += 1
            loc = action.replace("go to ", "", 1).strip()
            if loc in unique_locs:
                revisit_count += 1
            unique_locs.add(loc)
        if "nothing happens" in obs or step.get("action_error"):
            no_effect += 1

    nav_ratio = (navigation_steps / steps_used) if steps_used > 0 else 0.0

    # Strategy rubric heuristic
    if steps_used == 0:
        strategy = 5.0
        strat_band = "no data"
    elif (
        success
        and nav_ratio >= 0.4
        and revisit_count <= 1
        and no_effect <= 1
        and steps_used <= step_budget * 0.5
    ):
        strategy = 9.0
        strat_band = "excellent prioritization"
    elif nav_ratio >= 0.3 and revisit_count <= 2 and no_effect <= 2:
        strategy = 7.5
        strat_band = "mostly sensible"
    elif revisit_count <= 4 and no_effect <= 4:
        strategy = 6.0
        strat_band = "systematic but poorly prioritized"
    elif revisit_count <= 6:
        strategy = 4.0
        strat_band = "partially systematic"
    else:
        strategy = 2.0
        strat_band = "chaotic"
    strategy = _clamp(strategy)

    # Reasoning features
    reasonings = [str(step.get("reasoning", "")).strip() for step in trajectory_steps]
    nonempty_reasonings = [r for r in reasonings if r]
    reasoning_coverage = (
        (len(nonempty_reasonings) / steps_used) if steps_used > 0 else 0.0
    )

    # Goal keywords from task_text plus a small default set
    base_goal_keywords = {"goal", "task", "book", "desklamp", "light", "examine"}
    for tok in _normalize_text(task_text).split():
        if len(tok) > 3:
            base_goal_keywords.add(tok)
    goal_reference_hits = sum(
        1
        for r in nonempty_reasonings
        if any(k in r.lower() for k in base_goal_keywords)
    )
    goal_reference_rate = (
        (goal_reference_hits / len(nonempty_reasonings)) if nonempty_reasonings else 0.0
    )

    # Observation grounding: approximate by matching observation tokens of each step
    grounding_hits = 0
    for step, r in zip(trajectory_steps, reasonings):
        if not r:
            continue
        obs_tokens = set(_normalize_text(str(step.get("observation", ""))).split())
        if not obs_tokens:
            continue
        r_tokens = set(_normalize_text(r).split())
        if r_tokens & obs_tokens:
            grounding_hits += 1
    observation_grounding_rate = (
        grounding_hits / len(nonempty_reasonings) if nonempty_reasonings else 0.0
    )

    # Repetition rate: crude duplicate detection
    normalized_rs = [_normalize_text(r) for r in nonempty_reasonings]
    unique_rs = set(normalized_rs)
    repetition_rate = (
        1.0 - (len(unique_rs) / len(normalized_rs)) if normalized_rs else 0.0
    )

    # Reasoning rubric heuristic
    if (
        reasoning_coverage >= 0.8
        and goal_reference_rate >= 0.5
        and observation_grounding_rate >= 0.4
        and repetition_rate <= 0.15
    ):
        reasoning_score = 9.0
        reasoning_band = "goal-referenced, grounded, adaptive"
    elif (
        reasoning_coverage >= 0.6
        and goal_reference_rate >= 0.35
        and observation_grounding_rate >= 0.25
        and repetition_rate <= 0.25
    ):
        reasoning_score = 7.5
        reasoning_band = "generally grounded with goal references"
    elif (
        reasoning_coverage >= 0.4
        and goal_reference_rate >= 0.2
        and repetition_rate <= 0.35
    ):
        reasoning_score = 6.0
        reasoning_band = "coherent but shallow"
    elif reasoning_coverage >= 0.2:
        reasoning_score = 4.0
        reasoning_band = "generic/limited grounding"
    else:
        reasoning_score = 2.0
        reasoning_band = "no goal awareness"
    reasoning_score = _clamp(reasoning_score)

    # Overall weighted (using new balanced weights)
    overall = compute_weighted_overall(
        correctness, efficiency, strategy, reasoning_score
    )
    quick = round(overall, 2)  # quick rating definition: reuse overall

    features = {
        "unique_locations_visited": len(unique_locs),
        "revisit_count": revisit_count,
        "invalid_or_noeffect_actions_count": no_effect,
        "exploration_ratio": round(nav_ratio, 2),
        "navigation_steps": navigation_steps,
        "reasoning_coverage": round(reasoning_coverage, 2),
        "goal_reference_rate": round(goal_reference_rate, 2),
        "observation_grounding_rate": round(observation_grounding_rate, 2),
        "repetition_rate": round(repetition_rate, 2),
    }
    notes = {
        "correctness_reason": correctness_reason,
        "efficiency_ratio": round(ratio, 2),
        "strategy_band": strat_band,
        "reasoning_band": reasoning_band,
    }

    return TrajectoryEval(
        task_id=str(task_id),
        success=success,
        steps=steps_used,
        correctness=round(correctness, 2),
        efficiency=round(efficiency, 2),
        strategy=round(strategy, 2),
        reasoning=round(reasoning_score, 2),
        overall=round(overall, 2),
        quick=round(quick, 2),
        features=features,
        notes=notes,
        failure_patterns=failure_patterns,
    )


def print_task_eval(
    eval: TrajectoryEval, *, task_text: str, step_budget: int = STEP_BUDGET
) -> str:
    lines: List[str] = []
    lines.append(f"--- GREEN AGENT EVALUATION (Task {eval.task_id}) ---")
    lines.append("")
    lines.append(f"Task Success: {'YES' if eval.success else 'NO'}")
    lines.append(f"Steps Used: {eval.steps} / {step_budget}")
    lines.append("")
    lines.append("Correctness / Task Completion")
    lines.append(f"  Reasoning: {eval.notes.get('correctness_reason', '')}")
    lines.append(
        f"  Calculation: success == {eval.success} → score = {eval.correctness:.1f} / 10"
    )
    lines.append("")
    lines.append("Efficiency (Step Budget)")
    lines.append(
        f"  Reasoning: {'Env succeeded' if eval.success else 'Env failed'}; step budget model."
    )
    lines.append("  Calculation:")
    ratio = eval.notes.get("efficiency_ratio", 0.0)
    lines.append(f"    ratio = {eval.steps} / {step_budget} = {ratio:.2f}")
    lines.append(f"    score = 10 × (1 − {ratio:.2f}) = {eval.efficiency:.1f} / 10")
    lines.append("")
    lines.append("Strategy Quality")
    llm_evaluated = eval.features.get("llm_evaluated", False)
    if llm_evaluated:
        lines.append("  [LLM Evaluated]")
        strategy_rationale = eval.notes.get("strategy_rationale", "")
        if strategy_rationale:
            lines.append(f"  Assessment: {strategy_rationale}")
    else:
        lines.append(
            "  Features: unique_locs={}; revisits={}; no_effect={}; nav_ratio={}".format(
                eval.features.get("unique_locations_visited", 0),
                eval.features.get("revisit_count", 0),
                eval.features.get("invalid_or_noeffect_actions_count", 0),
                eval.features.get("exploration_ratio", 0.0),
            )
        )
        lines.append(f"  Reasoning: {eval.notes.get('strategy_band', '')}.")
    lines.append(f"  Score: {eval.strategy:.1f} / 10")
    lines.append("")
    lines.append("Reasoning Quality")
    if llm_evaluated:
        lines.append("  [LLM Evaluated]")
        reasoning_rationale = eval.notes.get("reasoning_rationale", "")
        if reasoning_rationale:
            lines.append(f"  Assessment: {reasoning_rationale}")
    else:
        lines.append(
            "  Features: coverage={}; goal_ref={}; grounding={}; repetition={}".format(
                eval.features.get("reasoning_coverage", 0.0),
                eval.features.get("goal_reference_rate", 0.0),
                eval.features.get("observation_grounding_rate", 0.0),
                eval.features.get("repetition_rate", 0.0),
            )
        )
        lines.append(f"  Reasoning: {eval.notes.get('reasoning_band', '')}.")
    lines.append(f"  Score: {eval.reasoning:.1f} / 10")
    lines.append("")
    lines.append(f"Overall Rating (weighted): {eval.overall:.1f} / 10")
    lines.append(
        f"  Breakdown: task_completion*{WEIGHT_TASK_COMPLETION:.2f} + efficiency*{WEIGHT_EFFICIENCY:.2f} + strategy*{WEIGHT_STRATEGY:.2f} + reasoning*{WEIGHT_REASONING:.2f}"
    )
    lines.append(
        "             {:.1f}*{:.2f} + {:.1f}*{:.2f} + {:.1f}*{:.2f} + {:.1f}*{:.2f} = {:.2f}".format(
            eval.correctness,
            WEIGHT_TASK_COMPLETION,
            eval.efficiency,
            WEIGHT_EFFICIENCY,
            eval.strategy,
            WEIGHT_STRATEGY,
            eval.reasoning,
            WEIGHT_REASONING,
            eval.overall,
        )
    )

    # Report detected failure patterns
    if eval.failure_patterns:
        lines.append("")
        lines.append("Detected Failure Patterns:")
        for fp in eval.failure_patterns:
            severity_marker = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                fp.severity, "•"
            )
            lines.append(
                f"  {severity_marker} {fp.name} ({fp.severity}): {fp.description}"
            )
            for ev in fp.evidence:
                lines.append(f"      Evidence: {ev}")

    return "\n".join(lines)


def aggregate_evals(
    evals: List[TrajectoryEval], step_budget: int = STEP_BUDGET
) -> Dict[str, Any]:
    total = len(evals)
    success_count = sum(1 for e in evals if e.success)
    success_rate = (success_count / total) if total else 0.0
    mean_steps = _safe_mean([e.steps for e in evals])
    mean_quick = _safe_mean([e.quick for e in evals])
    mean_overall = _safe_mean([e.overall for e in evals])
    mean_strategy = _safe_mean([e.strategy for e in evals])
    mean_reasoning = _safe_mean([e.reasoning for e in evals])

    mean_step_ratio = mean_steps / float(step_budget) if step_budget > 0 else 1.0
    efficiency_batch = _clamp(10.0 * (1.0 - mean_step_ratio))
    correctness_batch = _clamp(10.0 * success_rate)

    return {
        "total": total,
        "success_count": success_count,
        "success_rate": success_rate,
        "mean_steps": mean_steps,
        "mean_quick": mean_quick,
        "mean_overall": mean_overall,
        "efficiency_batch": efficiency_batch,
        "correctness_batch": correctness_batch,
        "mean_strategy": mean_strategy,
        "mean_reasoning": mean_reasoning,
    }


def print_batch_summary(
    *,
    evals: List[TrajectoryEval],
    white_label: str,
    step_budget: int = STEP_BUDGET,
) -> str:
    summary = aggregate_evals(evals, step_budget=step_budget)
    lines: List[str] = []
    lines.append(f"--- GREEN AGENT ASSESSMENT (5 tasks, {white_label}) ---")
    lines.append("")
    lines.append("Aggregate")
    lines.append(
        f"- Success rate: {summary['success_count']}/{summary['total']} ({summary['success_rate'] * 100:.0f}%)"
    )
    lines.append(f"- Mean steps: {summary['mean_steps']:.1f} / {step_budget}")
    lines.append(f"- Quick rating (avg): {summary['mean_quick']:.1f} / 10")
    lines.append(f"- Overall rating (avg): {summary['mean_overall']:.1f} / 10")
    lines.append(f"- Efficiency (batch): {summary['efficiency_batch']:.1f} / 10")
    lines.append(f"- Correctness (batch): {summary['correctness_batch']:.1f} / 10")
    lines.append("")
    lines.append("Per-criterion (avg)")
    lines.append(f"- Strategy Quality: {summary['mean_strategy']:.1f} / 10")
    lines.append(f"- Reasoning Quality: {summary['mean_reasoning']:.1f} / 10")
    lines.append("")
    lines.append("Per-task (brief)")
    for e in evals:
        lines.append(
            f"- Task {e.task_id}: success={'Yes' if e.success else 'No'}, steps={e.steps}, overall={e.overall:.1f}"
        )
    return "\n".join(lines)
