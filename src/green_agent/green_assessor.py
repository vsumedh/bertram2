"""Assessment utilities for green agent multi-task evaluation."""

from __future__ import annotations

import logging
import re
import statistics
from collections import Counter
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Sequence, Optional, Tuple, TYPE_CHECKING

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..utils.textworld_env import ExpertTrajectory


STEP_BUDGET = 50


# Default scoring weights (can be overridden by rubric)
DEFAULT_WEIGHTS = {
    "task_completion": 0.30,
    "efficiency": 0.20,
    "strategy": 0.25,
    "reasoning": 0.25,
}

# Module-level constants for backward compatibility
WEIGHT_TASK_COMPLETION = DEFAULT_WEIGHTS["task_completion"]
WEIGHT_EFFICIENCY = DEFAULT_WEIGHTS["efficiency"]
WEIGHT_STRATEGY = DEFAULT_WEIGHTS["strategy"]
WEIGHT_REASONING = DEFAULT_WEIGHTS["reasoning"]
WEIGHT_CORRECTNESS = WEIGHT_TASK_COMPLETION  # Alias


def get_weights(rubric: Optional[Any] = None) -> Dict[str, float]:
    """Get evaluation weights, preferring rubric if provided.

    Args:
        rubric: Optional EvaluationRubric instance with criteria weights.

    Returns:
        Dictionary mapping criterion names to weights.
    """
    if rubric is not None and hasattr(rubric, "criteria") and rubric.criteria:
        weights = {}
        criteria_map = {
            "task_completion": "task_completion",
            "efficiency": "efficiency",
            "strategy": "strategy_quality",
            "reasoning": "reasoning_quality",
        }
        for key, rubric_key in criteria_map.items():
            criterion = rubric.criteria.get(rubric_key)
            if criterion and hasattr(criterion, "weight"):
                weights[key] = criterion.weight
            else:
                weights[key] = DEFAULT_WEIGHTS[key]
        return weights
    return DEFAULT_WEIGHTS.copy()


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
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute overall score with configurable weights.

    Args:
        correctness: Task completion score (0-10)
        efficiency: Efficiency score (0-10)
        strategy: Strategy quality score (0-10)
        reasoning: Reasoning quality score (0-10)
        weights: Optional custom weights dict. Uses DEFAULT_WEIGHTS if None.

    Returns:
        Weighted overall score (0-10)
    """
    w = weights or DEFAULT_WEIGHTS
    return round(
        correctness * w["task_completion"]
        + efficiency * w["efficiency"]
        + strategy * w["strategy"]
        + reasoning * w["reasoning"],
        2,
    )


def _compute_expert_strategy_score(
    agent_actions: List[str],
    expert_actions: List[str],
) -> Tuple[float, str, Dict[str, Any]]:
    """Compute strategy score based on expert trajectory comparison.

    Args:
        agent_actions: List of actions taken by the agent
        expert_actions: List of baseline actions (ground truth or handcoded expert)

    Returns:
        Tuple of (score, band_description, features_dict)
    """

    def normalize_action(action: str) -> str:
        """Normalize action for comparison (lowercase, collapse whitespace, remove instance numbers)."""
        action = " ".join(action.lower().split())
        return re.sub(r"\s+\d+", " N", action)  # "take book 1" -> "take book N"

    agent_norm = [normalize_action(a) for a in agent_actions]
    expert_norm = [normalize_action(a) for a in expert_actions]

    # 1. Critical action coverage (goal-advancing actions)
    critical_verbs = {"take", "move", "put", "clean", "heat", "cool", "use"}
    expert_critical = [a for a in expert_norm if a.split()[0] in critical_verbs]
    agent_critical = [a for a in agent_norm if a.split()[0] in critical_verbs]

    critical_found = len(set(agent_critical) & set(expert_critical))
    critical_total = len(set(expert_critical))
    critical_coverage = critical_found / critical_total if critical_total else 1.0

    # 2. Path deviation (extra actions beyond expert)
    extra_actions = len(agent_actions) - len(expert_actions)
    deviation_penalty = min(extra_actions / 10.0, 1.0) if extra_actions > 0 else 0.0

    # 3. Location overlap (navigation efficiency)
    def extract_locations(actions: List[str]) -> List[str]:
        return [a.replace("go to ", "") for a in actions if a.startswith("go to ")]

    agent_locs = set(extract_locations(agent_norm))
    expert_locs = set(extract_locations(expert_norm))
    unnecessary_locs = len(agent_locs - expert_locs)
    loc_penalty = min(unnecessary_locs / 5.0, 1.0)

    # Composite score
    base_score = 10.0 * critical_coverage
    score = _clamp(base_score - (deviation_penalty * 2) - (loc_penalty * 2))

    # Determine band
    if score >= 8.0:
        band = "near-baseline path"
    elif score >= 6.0:
        band = "good with minor deviations"
    elif score >= 4.0:
        band = "significant detours from baseline"
    else:
        band = "fundamentally different approach"

    features = {
        "critical_coverage": round(critical_coverage, 2),
        "extra_actions": extra_actions,
        "unnecessary_locations": unnecessary_locs,
        "expert_steps": len(expert_actions),
    }

    return score, band, features


def evaluate_trajectory(
    *,
    task_id: str,
    task_text: str,
    step_budget: int,
    trajectory_steps: List[Dict[str, Any]],
    env_success: bool,
    expert_actions: List[str],
) -> TrajectoryEval:
    """Evaluate a trajectory against the expert baseline.

    Args:
        task_id: Task identifier
        task_text: Task goal description
        step_budget: Maximum allowed steps
        trajectory_steps: List of trajectory step dictionaries
        env_success: Whether task was completed successfully
        expert_actions: List of baseline actions (ground truth or handcoded expert)

    Returns:
        TrajectoryEval with complete evaluation results

    Raises:
        ValueError: If expert_actions is empty or invalid
    """
    # Validate expert_actions - this is required, no fallbacks
    if not expert_actions or len(expert_actions) == 0:
        raise ValueError(
            "expert_actions is required and must not be empty. "
            "The baseline trajectory must be provided (ground truth or handcoded expert)."
        )

    steps_used = len(trajectory_steps)
    success = bool(env_success)
    expert_steps = len(expert_actions)

    # Detect failure patterns for diagnostic purposes
    failure_patterns = detect_failure_patterns(trajectory_steps, success, task_text)

    # Task Completion (was: Correctness)
    correctness = 10.0 if success else 0.0
    correctness_reason = (
        "Env reports success (goal state reached)."
        if success
        else "Env reports failure."
    )

    # Efficiency - expert-relative only (no fallbacks)
    if success:
        # Baseline-relative efficiency: 1.0x baseline = 10, 2.0x baseline = 5, 3.0x+ = ~0
        overhead_ratio = steps_used / expert_steps
        # Use gentle decay: 10 * (baseline / actual)
        efficiency = _clamp(10.0 * (1.0 / overhead_ratio))
        ratio = overhead_ratio  # Store for notes
    else:
        ratio = steps_used / float(expert_steps)
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

    # Strategy scoring: expert-based only (no heuristic fallback)
    agent_action_list = [str(s.get("action", "")) for s in trajectory_steps]
    strategy, strat_band, expert_strategy_features = _compute_expert_strategy_score(
        agent_action_list, expert_actions
    )
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

    # Diagnostic logging for reasoning quality analysis
    _logger.debug(
        f"[Reasoning Metrics] task_id={task_id}, steps={steps_used}"
    )
    _logger.debug(
        f"  coverage={reasoning_coverage:.3f}, goal_ref={goal_reference_rate:.3f}, "
        f"grounding={observation_grounding_rate:.3f}, repetition={repetition_rate:.3f}"
    )
    _logger.debug(f"  unique_reasonings={len(unique_rs)}/{len(normalized_rs)}")

    # Log duplicates if repetition is high
    if repetition_rate > 0.10:
        counts = Counter(normalized_rs)
        duplicates = [
            (r[:60] + "..." if len(r) > 60 else r, c)
            for r, c in counts.most_common(3)
            if c > 1
        ]
        if duplicates:
            _logger.debug(f"  top_duplicates: {duplicates}")

    # Log goal keywords and which reasonings missed them
    if goal_reference_rate < 0.9:
        missed_indices = [
            i
            for i, r in enumerate(nonempty_reasonings)
            if not any(k in r.lower() for k in base_goal_keywords)
        ][:5]
        _logger.debug(f"  missed_goal_ref_at_steps: {missed_indices}")
        _logger.debug(f"  goal_keywords: {list(base_goal_keywords)[:10]}")

    # Reasoning rubric heuristic
    # Exceptional band for perfect scores (stricter thresholds)
    if (
        reasoning_coverage >= 0.95
        and goal_reference_rate >= 0.85
        and observation_grounding_rate >= 0.75
        and repetition_rate <= 0.05
    ):
        reasoning_score = 10.0
        reasoning_band = "exceptional: consistent goal awareness, grounding, uniqueness"
    elif (
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
        # Expert-based features (from ground truth or handcoded expert baseline)
        "expert_steps": expert_steps,
        "step_overhead": steps_used - expert_steps,
        "expert_relative_efficiency": round(steps_used / expert_steps, 2),
        # Store expert actions for logging/verification
        "expert_actions": list(expert_actions),
    }
    # Merge expert strategy features
    features.update(expert_strategy_features)

    notes = {
        "correctness_reason": correctness_reason,
        "efficiency_ratio": round(ratio, 2),
        "strategy_band": strat_band,
        "reasoning_band": reasoning_band,
        "expert_evaluated": True,  # Always true now - expert trajectory is required
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
    expert_steps = eval.features.get("expert_steps")
    lines.append("Efficiency (Expert-Relative)")
    baseline_source = eval.notes.get("baseline_source", "handcoded_expert") if isinstance(eval.notes, dict) else "handcoded_expert"
    baseline_label = "Ground Truth" if baseline_source == "ground_truth" else "Handcoded Expert"
    lines.append(
        f"  Reasoning: {'Env succeeded' if eval.success else 'Env failed'}; comparing to {expert_steps} baseline steps ({baseline_label})."
    )
    lines.append("  Calculation:")
    ratio = eval.notes.get("efficiency_ratio", 0.0)
    lines.append(
        f"    overhead_ratio = {eval.steps} / {expert_steps} = {ratio:.2f}x baseline"
    )
    lines.append(
        f"    score = 10 × (1.0 / {ratio:.2f}) = {eval.efficiency:.1f} / 10"
    )
    step_overhead = eval.features.get("step_overhead", 0)
    lines.append(f"  Agent overhead: {step_overhead} extra steps beyond baseline")
    # Log expert actions for verification
    expert_actions = eval.features.get("expert_actions")
    if expert_actions:
        lines.append(f"  Expert actions: {expert_actions}")
    lines.append("")
    lines.append("Strategy Quality")
    llm_evaluated = eval.features.get("llm_evaluated", False)
    if llm_evaluated:
        lines.append("  [LLM Evaluated]")
        strategy_rationale = eval.notes.get("strategy_rationale", "")
        if strategy_rationale:
            lines.append(f"  Assessment: {strategy_rationale}")
    else:
        # Expert-compared (always, since expert trajectory is required)
        lines.append("  [Expert-Compared]")
        lines.append(
            "  Features: critical_coverage={}; extra_actions={}; unnecessary_locs={}".format(
                eval.features.get("critical_coverage", 0.0),
                eval.features.get("extra_actions", 0),
                eval.features.get("unnecessary_locations", 0),
            )
        )
        lines.append(f"  Assessment: {eval.notes.get('strategy_band', '')}.")
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


async def evaluate(
    trajectory_steps: List[Dict[str, Any]],
    *,
    goal: str,
    success: bool,
    expert_actions: List[str],
    task_id: str = "0",
    step_budget: int = STEP_BUDGET,
    expert_trajectory: Optional["ExpertTrajectory"] = None,
    use_llm: bool = True,
    baseline_source: str = "handcoded_expert",
) -> TrajectoryEval:
    """
    Unified evaluation entry point combining heuristic and LLM assessment.

    This is the primary evaluation function that:
    1. Computes heuristic scores (correctness, efficiency, strategy, reasoning)
    2. Optionally enhances strategy/reasoning with LLM evaluation
    3. If expert_trajectory provided, uses LLM-as-judge to compare trajectories

    Args:
        trajectory_steps: List of trajectory step dictionaries
        goal: Task goal description
        success: Whether task was completed successfully
        expert_actions: List of baseline actions (ground truth or handcoded expert)
        task_id: Task identifier for logging
        step_budget: Maximum allowed steps
        expert_trajectory: Optional full expert trajectory for detailed comparison
        use_llm: Whether to attempt LLM evaluation for strategy/reasoning
        baseline_source: Source of baseline ("ground_truth" or "handcoded_expert")

    Returns:
        TrajectoryEval with complete evaluation results

    Raises:
        ValueError: If expert_actions is empty or invalid
    """
    import logging

    logger = logging.getLogger(__name__)

    # Get heuristic evaluation (always needed for correctness/efficiency)
    heuristic_eval = evaluate_trajectory(
        task_id=task_id,
        task_text=goal,
        step_budget=step_budget,
        trajectory_steps=trajectory_steps,
        env_success=success,
        expert_actions=expert_actions,
    )
    
    # Add baseline source to notes
    if isinstance(heuristic_eval.notes, dict):
        heuristic_eval.notes["baseline_source"] = baseline_source
    
    if not use_llm:
        return heuristic_eval

    # Try LLM enhancement for reasoning/strategy
    llm_scores = None

    try:
        from .evaluator import evaluate_with_llm

        # Standard LLM evaluation for reasoning/strategy
        llm_scores = await evaluate_with_llm(
            goal=goal,
            trajectory=trajectory_steps,
            success=success,
            step_budget=step_budget,
        )

        if llm_scores:
            logger.info(
                f"[LLM Scores] Received: reasoning={llm_scores['reasoning_score']:.1f}, "
                f"strategy={llm_scores['strategy_score']:.1f}"
            )
            logger.debug(
                f"[LLM Scores] Rationales: reasoning='{llm_scores['reasoning_rationale'][:100]}...', "
                f"strategy='{llm_scores['strategy_rationale'][:100]}...'"
            )

            # Build enhanced features with expert comparison data
            enhanced_features = {
                **heuristic_eval.features,
                "llm_evaluated": True,
            }
            enhanced_notes = {
                **(
                    heuristic_eval.notes
                    if isinstance(heuristic_eval.notes, dict)
                    else {}
                ),
                "reasoning_rationale": llm_scores["reasoning_rationale"],
                "strategy_rationale": llm_scores["strategy_rationale"],
                "notable_moments": llm_scores.get("notable_moments", []),
                "baseline_source": baseline_source,
            }

            # Log inputs immediately before final score computation
            logger.info(
                f"[Final Score] Computing weighted overall from: "
                f"correctness={heuristic_eval.correctness:.1f} (heuristic), "
                f"efficiency={heuristic_eval.efficiency:.1f} (heuristic), "
                f"strategy={llm_scores['strategy_score']:.1f} (LLM), "
                f"reasoning={llm_scores['reasoning_score']:.1f} (LLM)"
            )

            # Merge LLM scores with heuristic evaluation
            final_overall = compute_weighted_overall(
                heuristic_eval.correctness,
                heuristic_eval.efficiency,
                llm_scores["strategy_score"],
                llm_scores["reasoning_score"],
            )
            logger.info(
                f"[Final Score] Computed overall={final_overall:.2f} "
                f"(weights: correctness×{WEIGHT_TASK_COMPLETION}, efficiency×{WEIGHT_EFFICIENCY}, "
                f"strategy×{WEIGHT_STRATEGY}, reasoning×{WEIGHT_REASONING})"
            )

            return TrajectoryEval(
                task_id=heuristic_eval.task_id,
                success=heuristic_eval.success,
                steps=heuristic_eval.steps,
                correctness=heuristic_eval.correctness,
                efficiency=heuristic_eval.efficiency,
                strategy=round(llm_scores["strategy_score"], 2),
                reasoning=round(llm_scores["reasoning_score"], 2),
                overall=final_overall,
                quick=heuristic_eval.quick,
                features=enhanced_features,
                notes=enhanced_notes,
                failure_patterns=heuristic_eval.failure_patterns,
            )
    except Exception as e:
        logger.warning(f"LLM evaluation failed, using heuristic only: {e}")

    return heuristic_eval


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

    # Simply average efficiency/correctness from individual evaluations
    # instead of recalculating (which was causing discrepancy with expert-relative scores)
    efficiency_batch = _safe_mean([e.efficiency for e in evals])
    correctness_batch = _safe_mean([e.correctness for e in evals])

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
