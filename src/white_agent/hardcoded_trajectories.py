"""Pre-recorded expert trajectories for ALFWorld tasks.

Each trajectory contains:
- task_type: The ALFWorld task type
- object: The target object being manipulated
- target: The destination/tool
- actions: List of expert actions
- reasoning_templates: Templates for generating contextual reasoning

This module also provides observation-aware strategy execution for adaptive agents.
"""

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional, Set


# Deterministic agent profiles for observation-aware action selection + reasoning.
# NOTE: No backwards-compatibility is provided; callers must use these keys.
#
# Profile Calibration Targets (relative to pre-recorded expert trajectory):
# - expert:       1.0x  (follows pre-recorded optimal trajectory, exceptional reasoning)
# - competent:    2.0-3.0x  (good priorities, occasional detours)
# - overthinker:  2.5-3.5x  (good priorities, extra verification actions)
# - lucky_guesser: 1.0-1.5x strategy, 1-3/10 reasoning (good actions, poor explanations)
# - novice:       4.0-6.0x  (random priorities, many inefficiencies)
#
AGENT_PROFILES: Dict[str, Dict[str, Any]] = {
    # Expert profile: Follows pre-recorded optimal trajectory with exceptional reasoning.
    # Uses prefer_prerecorded_trajectory to match expert actions exactly when valid.
    # Target: Perfect strategy score (10.0) and exceptional reasoning score (10.0)
    "expert": {
        "description": "Optimal: follows expert trajectory, exceptional reasoning (1.0x).",
        # Strategy knobs - use pre-recorded trajectory for optimal actions
        "search_priority_noise_mode": "none",
        "detour_every_n_steps": 0,
        "detour_location_type_priority": [],
        "extra_look_after_goto": False,
        "inventory_check_every_n_steps": 0,
        "extra_open_close_every_n_steps": 0,
        "forget_visited_every_n_steps": 0,
        "premature_navigate_every_n_steps": 0,
        "stall_look_every_n_steps": 0,
        "prefer_prerecorded_trajectory": True,  # Follow optimal trajectory
        "use_ground_truth": False,
        # Reasoning knobs - maximum quality for 10.0 score
        "reasoning_style": "exceptional",  # New style for highest quality
        "reasoning_coverage_ratio": 1.0,
        "goal_reference_every_n_steps": 1,
        "observation_grounding_every_n_steps": 1,
        "forced_repetition_phrase": None,
        "include_source_location_in_reasoning": True,
        "include_visible_objects_in_reasoning": True,  # New: adds observation grounding
    },
    # Competent profile: Good but not optimal search, minor inefficiencies.
    # Target: 2.5-3.5x ground truth steps
    "competent": {
        "description": "Good behavior with minor inefficiencies (2.5-3.5x).",
        "search_priority_noise_mode": "swap_first",  # Occasionally checks 2nd-best location first
        "detour_every_n_steps": 10,  # Reduced from 7 for better efficiency
        "detour_location_type_priority": ["drawer", "cabinet"],
        "extra_look_after_goto": False,
        "inventory_check_every_n_steps": 0,
        "extra_open_close_every_n_steps": 0,
        "forget_visited_every_n_steps": 0,
        "premature_navigate_every_n_steps": 0,
        "stall_look_every_n_steps": 0,
        "prefer_prerecorded_trajectory": False,
        "use_ground_truth": False,
        "reasoning_style": "medium",
        "reasoning_coverage_ratio": 0.85,
        "goal_reference_every_n_steps": 2,
        "observation_grounding_every_n_steps": 2,
        "forced_repetition_phrase": None,
        "include_source_location_in_reasoning": False,
    },
    # Overthinker profile: High reasoning quality but extra verification actions.
    # Target: 3.0-4.0x ground truth steps
    "overthinker": {
        "description": "Thorough but slow: extra checks reduce efficiency (3-4x).",
        "search_priority_noise_mode": "none",  # Good priorities
        "detour_every_n_steps": 0,
        "detour_location_type_priority": [],
        "extra_look_after_goto": True,  # Adds look after each goto
        "inventory_check_every_n_steps": 4,  # Periodic inventory checks
        "extra_open_close_every_n_steps": 5,  # Opens/closes containers
        "forget_visited_every_n_steps": 0,
        "premature_navigate_every_n_steps": 0,
        "stall_look_every_n_steps": 0,
        "prefer_prerecorded_trajectory": False,
        "use_ground_truth": False,
        "reasoning_style": "high",
        "reasoning_coverage_ratio": 1.0,
        "goal_reference_every_n_steps": 1,
        "observation_grounding_every_n_steps": 1,
        "forced_repetition_phrase": None,
        "include_source_location_in_reasoning": True,
    },
    # Lucky guesser: Good strategy but terrible reasoning explanations.
    # Target: 1.5-2.0x steps but 1-3/10 reasoning score
    "lucky_guesser": {
        "description": "Good actions but poor reasoning (1.5-2x steps, low reasoning).",
        "search_priority_noise_mode": "none",
        "detour_every_n_steps": 0,
        "detour_location_type_priority": [],
        "extra_look_after_goto": False,
        "inventory_check_every_n_steps": 0,
        "extra_open_close_every_n_steps": 0,
        "forget_visited_every_n_steps": 0,
        "premature_navigate_every_n_steps": 0,
        "stall_look_every_n_steps": 0,
        "prefer_prerecorded_trajectory": False,
        "use_ground_truth": False,
        "reasoning_style": "low",
        "reasoning_coverage_ratio": 0.25,
        "goal_reference_every_n_steps": 0,
        "observation_grounding_every_n_steps": 0,
        "forced_repetition_phrase": "Proceeding.",
        "include_source_location_in_reasoning": False,
    },
    # Novice profile: Poor search strategy, many inefficiencies.
    # Target: 4.0-6.0x ground truth steps
    "novice": {
        "description": "Inefficient: random search, many detours (4-6x).",
        "search_priority_noise_mode": "shuffle",  # Random search order
        "detour_every_n_steps": 1,  # Very frequent detours (every step)
        "detour_location_type_priority": [
            "bed",
            "sofa",
            "armchair",
            "drawer",
            "cabinet",
        ],
        "extra_look_after_goto": True,
        "inventory_check_every_n_steps": 5,
        "extra_open_close_every_n_steps": 3,
        "forget_visited_every_n_steps": 4,  # Sometimes revisits locations
        "premature_navigate_every_n_steps": 0,
        "stall_look_every_n_steps": 4,
        "prefer_prerecorded_trajectory": False,
        "use_ground_truth": False,
        "reasoning_style": "low",
        "reasoning_coverage_ratio": 0.4,
        "goal_reference_every_n_steps": 0,
        "observation_grounding_every_n_steps": 0,
        "forced_repetition_phrase": "Let me explore.",
        "include_source_location_in_reasoning": False,
    },
}


# Module-level RNG for non-seeded randomness (doesn't affect global state)
_module_rng = random.Random()


def _generate_low_quality_reasoning(*, step: int) -> str:
    """Generate minimal/generic reasoning (deterministic by step)."""
    templates = [
        "Proceeding.",
        "Next step.",
        "Continuing.",
        "Moving on.",
        "",  # sometimes no reasoning
    ]
    return templates[step % len(templates)]


def _generate_medium_quality_reasoning(*, action: str, step: int) -> str:
    """Generate action-aware but generic reasoning (deterministic by step)."""
    action_lower = action.lower()

    if action_lower.startswith("go to "):
        templates = [
            "Moving to check another location.",
            "Going to a new area.",
            "Navigating to the next spot.",
        ]
        return templates[step % len(templates)]
    elif action_lower.startswith("take "):
        templates = [
            "Picking up an item.",
            "Grabbing something.",
            "Taking this object.",
        ]
        return templates[step % len(templates)]
    elif action_lower.startswith("move ") or action_lower.startswith("put "):
        templates = [
            "Placing the item.",
            "Putting it down.",
            "Moving the object.",
        ]
        return templates[step % len(templates)]
    elif action_lower.startswith("open "):
        return "Opening to check inside."
    elif action_lower.startswith("close "):
        return "Closing after checking."
    elif action_lower.startswith("clean "):
        return "Cleaning the item."
    elif action_lower.startswith("heat "):
        return "Heating the item."
    elif action_lower.startswith("cool "):
        return "Cooling the item."
    elif action_lower.startswith("use "):
        return "Using this."
    elif action_lower == "look":
        return "Looking around."
    elif action_lower.startswith("examine "):
        return "Examining this."
    else:
        return f"Doing: {action}"


def _generate_high_quality_reasoning(
    action: str, step: int, goal: str, observation: str
) -> str:
    """Generate high-quality contextual reasoning for an action.

    This generates reasoning that:
    - References the goal/task keywords
    - References objects/locations from the observation
    - Varies by step number to avoid repetition
    """
    action_lower = action.lower()

    # Extract key elements from action
    if action_lower.startswith("go to "):
        location = action[6:]
        templates = [
            f"Navigating to {location} to search for the target object needed to complete the task.",
            f"Moving to {location} as part of systematic exploration to locate required items.",
            f"Proceeding to {location} to check for objects relevant to the goal.",
            f"Traveling to {location} to continue the search pattern toward task completion.",
            f"Heading to {location} which may contain items needed for the objective.",
        ]
        return templates[step % len(templates)]

    elif action_lower.startswith("take "):
        obj = action[5:].split(" from ")[0] if " from " in action else action[5:]
        templates = [
            f"Taking the {obj} which is required to complete the task objective.",
            f"Picking up {obj} as this item is needed for the goal.",
            f"Acquiring {obj} to progress toward task completion.",
            f"Grabbing the {obj} which matches what is needed for this task.",
        ]
        return templates[step % len(templates)]

    elif action_lower.startswith("move ") or action_lower.startswith("put "):
        parts = action.split(" to ")
        obj = parts[0].replace("move ", "").replace("put ", "")
        dest = parts[1] if len(parts) > 1 else "destination"
        templates = [
            f"Placing {obj} at {dest} to complete the placement requirement of the task.",
            f"Moving {obj} to {dest} as specified by the goal.",
            f"Depositing {obj} in {dest} to fulfill the task objective.",
            f"Putting {obj} at {dest} which satisfies the goal condition.",
        ]
        return templates[step % len(templates)]

    elif action_lower.startswith("open "):
        container = action[5:]
        templates = [
            f"Opening {container} to check for items needed for the task.",
            f"Opening {container} to search for relevant objects.",
            f"Accessing {container} to inspect its contents for goal-related items.",
        ]
        return templates[step % len(templates)]

    elif action_lower.startswith("close "):
        container = action[6:]
        templates = [
            f"Closing {container} after inspection, continuing the search.",
            f"Shutting {container} and moving on to check other locations.",
        ]
        return templates[step % len(templates)]

    elif action_lower.startswith("clean "):
        obj = action[6:].split(" with ")[0]
        templates = [
            f"Cleaning the {obj} as required by the task before placement.",
            f"Washing {obj} to satisfy the cleaning requirement of the goal.",
            f"Applying cleaning to {obj} as specified in the task objective.",
        ]
        return templates[step % len(templates)]

    elif action_lower.startswith("heat "):
        obj = action[5:].split(" with ")[0]
        templates = [
            f"Heating the {obj} as required by the task before placement.",
            f"Warming {obj} to satisfy the heating requirement of the goal.",
            f"Applying heat to {obj} as specified in the task objective.",
        ]
        return templates[step % len(templates)]

    elif action_lower.startswith("cool "):
        obj = action[5:].split(" with ")[0]
        templates = [
            f"Cooling the {obj} as required by the task before placement.",
            f"Chilling {obj} to satisfy the cooling requirement of the goal.",
            f"Applying cold to {obj} as specified in the task objective.",
        ]
        return templates[step % len(templates)]

    elif action_lower.startswith("use "):
        obj = action[4:]
        templates = [
            f"Using the {obj} to examine the held object and complete the look_at task.",
            f"Activating {obj} to fulfill the examination requirement of the goal.",
            f"Employing {obj} as the final step to complete the task objective.",
        ]
        return templates[step % len(templates)]

    elif action_lower == "look":
        templates = [
            "Surveying the environment to understand the current state and available objects.",
            "Looking around to gather information about nearby items and locations.",
            "Observing the surroundings to plan the next steps toward the goal.",
            "Examining the area to identify objects relevant to the task.",
        ]
        return templates[step % len(templates)]

    elif action_lower.startswith("examine "):
        obj = action[8:]
        templates = [
            f"Examining {obj} to gather more information about its contents or state.",
            f"Inspecting {obj} to determine if it contains relevant items.",
        ]
        return templates[step % len(templates)]

    else:
        return f"Executing action '{action}' to progress toward the task goal."


def generate_reasoning(
    action: str,
    step: int,
    goal: str,
    observation: str,
    profile: Dict[str, Any],
) -> str:
    """Generate deterministic reasoning according to the profile policy.

    Args:
        action: The action being taken
        step: Current step number
        goal: Task goal description
        observation: Current environment observation
        profile: Deterministic profile policy dict (see AGENT_PROFILES)

    Returns:
        Reasoning string (possibly empty) appropriate to the profile.
    """
    # ---- Coverage control (deterministic schedule) ----
    period = 10
    coverage = float(profile.get("reasoning_coverage_ratio", 1.0))
    coverage = max(0.0, min(1.0, coverage))
    nonempty_target = int(round(coverage * period))
    nonempty_target = max(0, min(period, nonempty_target))
    blanks = period - nonempty_target
    if blanks > 0 and (step % period) < blanks:
        return ""

    forced_phrase = profile.get("forced_repetition_phrase")
    if isinstance(forced_phrase, str) and forced_phrase:
        return forced_phrase

    style = str(profile.get("reasoning_style", "high")).lower()
    goal_every = int(profile.get("goal_reference_every_n_steps", 0) or 0)
    obs_every = int(profile.get("observation_grounding_every_n_steps", 0) or 0)

    include_goal = goal_every > 0 and (step % goal_every == 0)
    include_obs = obs_every > 0 and (step % obs_every == 0)

    target_obj = extract_target_object(goal) if include_goal else ""
    target_recep = extract_target_receptacle(goal) if include_goal else ""

    obs_data = parse_observation(observation) if include_obs else {}
    obs_loc = (obs_data.get("location") or "") if include_obs else ""
    obs_visible = (obs_data.get("visible_objects") or []) if include_obs else []

    # Build observation snippet with optional visible objects
    include_visible = profile.get("include_visible_objects_in_reasoning", False)
    obs_snippet = ""
    if obs_loc:
        obs_snippet = f"At {obs_loc}."
        if include_visible and obs_visible:
            # Add first 2 visible objects for grounding and uniqueness
            visible_str = ", ".join(obs_visible[:2])
            obs_snippet = f"At {obs_loc}, seeing {visible_str}."
    elif obs_visible:
        obs_snippet = f"Seeing {sorted(obs_visible)[0]}."

    # ---- Style-specific templates ----
    action_lower = action.lower().strip()
    goal_snippet = ""
    if target_obj and target_recep:
        goal_snippet = f"Goal: {target_obj} -> {target_recep}."
    elif target_obj:
        goal_snippet = f"Goal: {target_obj}."

    if style == "low":
        return _generate_low_quality_reasoning(step=step)

    if style == "medium":
        base = _generate_medium_quality_reasoning(action=action, step=step)
        parts = [base]
        if goal_snippet:
            parts.append(goal_snippet)
        if obs_snippet:
            parts.append(obs_snippet)
        return " ".join(p for p in parts if p)

    # High/Exceptional reasoning: explain why the action advances the task
    why = ""
    # Include source location for uniqueness when profile enables it
    include_source = profile.get("include_source_location_in_reasoning", False)
    is_exceptional = style == "exceptional"

    if action_lower.startswith("go to "):
        dest = action[6:].strip()
        if is_exceptional and target_obj and obs_loc:
            # Exceptional: include both source, destination, and reason
            why = f"Navigating from {obs_loc} to {dest} because {target_obj} may be located there."
        elif target_obj and include_source and obs_loc:
            why = f"Moving from {obs_loc} to {dest} to search for {target_obj}."
        elif target_obj:
            why = f"Going to {dest} to search for {target_obj}."
        elif include_source and obs_loc:
            why = f"Moving from {obs_loc} to {dest} to continue the search."
        else:
            why = f"Going to {dest} to continue the search."
    elif action_lower.startswith("take "):
        # Extract the actual item being taken for specificity
        item = (
            action[5:].split(" from ")[0].strip()
            if " from " in action
            else action[5:].strip()
        )
        if is_exceptional and target_obj:
            why = f"Taking {item} which matches the target {target_obj} needed for the task."
        elif target_obj:
            why = f"Taking {item} to progress toward handling {target_obj}."
        else:
            why = f"Taking {item} to make progress on the task."
    elif action_lower.startswith(("move ", "put ")):
        # Extract the actual item being placed
        verb = "move" if action_lower.startswith("move ") else "put"
        rest = action[len(verb) + 1 :]
        item = (
            rest.split(" to ")[0].strip()
            if " to " in rest
            else rest.split(" in ")[0].strip()
        )
        if is_exceptional and target_recep:
            why = f"Placing {item} at {target_recep} to complete the goal placement requirement."
        elif target_recep:
            why = f"Placing {item} at {target_recep} to satisfy the target location."
        else:
            why = f"Placing {item} to complete the placement step."
    elif action_lower.startswith(("clean ", "heat ", "cool ")):
        transform_verb = action_lower.split()[0]
        item = action[len(transform_verb) + 1 :].split(" with ")[0].strip()
        if is_exceptional:
            why = f"Applying {transform_verb} to {item} as required by the task before placing at {target_recep}."
        else:
            why = (
                f"Applying {transform_verb} transformation to {item} before placement."
            )
    elif action_lower.startswith("open "):
        container = action[5:].strip()
        if is_exceptional and target_obj:
            why = f"Opening {container} to check if {target_obj} is inside."
        else:
            why = f"Opening {container} to check contents systematically."
    elif action_lower.startswith("close "):
        container = action[6:].strip()
        why = f"Closing {container} after inspection to move on."
    elif action_lower.startswith("use "):
        tool = action[4:].strip()
        if is_exceptional and target_obj:
            why = f"Using {tool} to examine {target_obj} and complete the task."
        else:
            why = f"Using {tool} to complete the task."
    elif action_lower == "look":
        if is_exceptional and obs_loc:
            why = f"Surveying {obs_loc} to identify objects and plan the next action toward {target_obj}."
        elif include_source and obs_loc:
            why = f"Looking around at {obs_loc} to gather information for the next action."
        else:
            why = "Looking to gather information for the next action."
    elif action_lower.startswith("examine "):
        item = action[8:].strip()
        why = f"Examining {item} to verify relevant details."
    else:
        why = _generate_high_quality_reasoning(action, step, goal, observation)

    parts = []
    if goal_snippet:
        parts.append(goal_snippet)
    if obs_snippet:
        parts.append(obs_snippet)
    parts.append(why)
    return " ".join(p for p in parts if p)


# --- Strategy degradation functions ---


def _add_minor_inefficiencies(actions: List[str], seed: int = None) -> List[str]:
    """Insert occasional unnecessary actions (suboptimal strategy).

    Adds ~30% chance of extra 'look' after navigation.
    Uses a local Random instance to avoid affecting global state.
    """
    rng = random.Random(seed)

    result = []
    for action in actions:
        result.append(action)
        # 30% chance to insert unnecessary "look" after navigation
        if action.lower().startswith("go to ") and rng.random() < 0.3:
            result.append("look")
    return result


def _add_major_inefficiencies(actions: List[str], seed: int = None) -> List[str]:
    """Insert random detours and unnecessary exploration (poor strategy).

    Adds ~40% chance of random detours before important actions.
    Uses a local Random instance to avoid affecting global state.
    """
    rng = random.Random(seed)

    detour_locations = [
        "shelf 1",
        "drawer 1",
        "countertop 1",
        "cabinet 1",
        "desk 1",
        "bed 1",
        "sidetable 1",
    ]

    result = []
    for action in actions:
        # 40% chance to insert a random detour before each action
        if rng.random() < 0.4:
            detour = rng.choice(detour_locations)
            result.append(f"go to {detour}")
            result.append("look")
        result.append(action)
    return result


def degrade_trajectory(
    actions: List[str],
    quality: str = "optimal",
    seed: int = None,
) -> List[str]:
    """Degrade action sequence based on strategy quality level.

    Args:
        actions: List of expert actions
        quality: Strategy quality level ("optimal", "suboptimal", "poor")
        seed: Optional random seed for reproducibility

    Returns:
        Degraded action list (may be longer than original)
    """
    if quality == "optimal":
        return actions.copy()
    elif quality == "suboptimal":
        return _add_minor_inefficiencies(actions, seed)
    else:  # "poor"
        return _add_major_inefficiencies(actions, seed)


def get_profile(profile_name: str) -> Dict[str, Any]:
    """Get profile configuration by name.

    Args:
        profile_name: Name of the profile (expert, competent, novice, etc.)

    Returns:
        Deterministic profile policy dictionary.
    """
    return AGENT_PROFILES.get(profile_name, AGENT_PROFILES["expert"])


# Pre-recorded expert trajectories for all 20 tasks
EXPERT_TRAJECTORIES: Dict[int, Dict[str, Any]] = {
    # Task 0: look_at_obj_in_light-Book-None-DeskLamp-308
    0: {
        "task_type": "look_at_obj_in_light",
        "object": "book",
        "target": "desklamp",
        "room_signature": ["bed", "desk", "drawer", "shelf"],
        "actions": [
            "look",
            "go to shelf 1",
            "go to shelf 2",
            "go to shelf 3",
            "go to shelf 4",
            "go to shelf 5",
            "go to shelf 6",
            "go to drawer 1",
            "open drawer 1",
            "close drawer 1",
            "go to drawer 2",
            "look",
            "go to shelf 4",
            "go to drawer 3",
            "go to drawer 4",
            "open drawer 4",
            "close drawer 4",
            "go to drawer 5",
            "go to drawer 6",
            "open drawer 6",
            "close drawer 6",
            "go to desk 1",
            "go to desk 2",
            "go to bed 1",
            "take book 1 from bed 1",
            "go to desk 1",
            "use desklamp 1",
        ],
    },
    # Task 1: look_at_obj_in_light-Mug-None-DeskLamp-308
    1: {
        "task_type": "look_at_obj_in_light",
        "object": "mug",
        "target": "desklamp",
        "room_signature": ["bed", "desk", "drawer", "shelf"],
        "actions": [
            "look",
            "go to shelf 1",
            "go to shelf 2",
            "take mug 1 from shelf 2",
            "go to shelf 1",
            "go to shelf 3",
            "use desklamp 1",
        ],
    },
    # Task 2: look_at_obj_in_light-CD-None-DeskLamp-308
    2: {
        "task_type": "look_at_obj_in_light",
        "object": "cd",
        "target": "desklamp",
        "room_signature": ["bed", "desk", "drawer", "shelf", "safe", "garbagecan"],
        "actions": [
            "look",
            "go to shelf 1",
            "go to shelf 2",
            "go to shelf 3",
            "go to shelf 4",
            "go to shelf 5",
            "go to shelf 6",
            "go to safe 1",
            "open safe 1",
            "close safe 1",
            "go to garbagecan 1",
            "go to drawer 1",
            "open drawer 1",
            "close drawer 1",
            "go to drawer 2",
            "look",
            "go to safe 1",
            "open safe 1",
            "close safe 1",
            "go to drawer 3",
            "go to drawer 4",
            "open drawer 4",
            "close drawer 4",
            "go to drawer 5",
            "go to drawer 6",
            "open drawer 6",
            "close drawer 6",
            "go to desk 1",
            "go to desk 2",
            "take cd 1 from desk 2",
            "go to desk 1",
            "use desklamp 1",
        ],
    },
    # Task 3: look_at_obj_in_light-AlarmClock-None-DeskLamp-308
    3: {
        "task_type": "look_at_obj_in_light",
        "object": "alarmclock",
        "target": "desklamp",
        "room_signature": ["bed", "desk", "drawer", "shelf"],
        "actions": [
            "look",
            "go to shelf 1",
            "go to shelf 2",
            "go to shelf 3",
            "go to shelf 4",
            "go to shelf 5",
            "go to shelf 6",
            "go to desk 1",
            "go to desk 2",
            "take alarmclock 1 from desk 2",
            "go to desk 1",
            "use desklamp 1",
        ],
    },
    # Task 4: pick_and_place_simple-SaltShaker-None-Cabinet-10
    4: {
        "task_type": "pick_and_place_simple",
        "object": "saltshaker",
        "target": "cabinet",
        "room_signature": ["shelf", "cabinet", "countertop"],
        "actions": [
            "look",
            "go to shelf 1",
            "take saltshaker 1 from shelf 1",
            "go to cabinet 1",
            "move saltshaker 1 to cabinet 1",
        ],
    },
    # Task 5: pick_and_place_simple-Watch-None-Safe-219
    5: {
        "task_type": "pick_and_place_simple",
        "object": "watch",
        "target": "safe",
        "room_signature": ["sidetable", "shelf", "safe", "dresser"],
        "actions": [
            "look",
            "go to sidetable 1",
            "go to shelf 1",
            "go to shelf 2",
            "look",
            "go to shelf 7",
            "go to shelf 3",
            "go to shelf 4",
            "go to shelf 5",
            "go to shelf 6",
            "go to shelf 8",
            "go to shelf 9",
            "go to shelf 10",
            "go to shelf 11",
            "go to shelf 12",
            "go to safe 1",
            "open safe 1",
            "close safe 1",
            "go to dresser 1",
            "take watch 1 from dresser 1",
            "go to safe 1",
            "open safe 1",
            "move watch 1 to safe 1",
        ],
    },
    # Task 6: pick_and_place_simple-SaltShaker-None-Cabinet-10 (variant)
    6: {
        "task_type": "pick_and_place_simple",
        "object": "saltshaker",
        "target": "cabinet",
        "room_signature": ["shelf", "cabinet", "countertop"],
        "actions": [
            "look",
            "go to shelf 1",
            "go to shelf 2",
            "take saltshaker 1 from shelf 2",
            "go to cabinet 1",
            "move saltshaker 1 to cabinet 1",
        ],
    },
    # Task 7: pick_and_place_simple-Vase-None-Safe-219
    7: {
        "task_type": "pick_and_place_simple",
        "object": "vase",
        "target": "safe",
        "room_signature": ["sidetable", "shelf", "safe", "dresser", "drawer"],
        "actions": [
            "look",
            "go to sidetable 1",
            "go to shelf 1",
            "go to shelf 2",
            "look",
            "examine shelf 2",
            "look",
            "go to drawer 1",
            "open drawer 1",
            "close drawer 1",
            "go to shelf 3",
            "go to shelf 4",
            "go to shelf 5",
            "go to shelf 6",
            "go to shelf 7",
            "go to shelf 8",
            "go to shelf 9",
            "go to shelf 10",
            "go to shelf 11",
            "go to shelf 12",
            "go to safe 1",
            "open safe 1",
            "close safe 1",
            "go to dresser 1",
            "take vase 1 from dresser 1",
            "go to safe 1",
            "open safe 1",
            "move vase 1 to safe 1",
        ],
    },
    # Task 8: pick_clean_then_place_in_recep-Pan-None-CounterTop-10
    8: {
        "task_type": "pick_clean_then_place_in_recep",
        "object": "pan",
        "target": "countertop",
        "room_signature": ["toaster", "stoveburner", "sinkbasin"],
        "actions": [
            "look",
            "go to toaster 1",
            "go to stoveburner 1",
            "take pan 1 from stoveburner 1",
            "go to sinkbasin 1",
            "clean pan 1 with sinkbasin 1",
        ],
    },
    # Task 9: pick_clean_then_place_in_recep-SoapBar-None-CounterTop-424
    9: {
        "task_type": "pick_clean_then_place_in_recep",
        "object": "soapbar",
        "target": "countertop",
        "room_signature": ["toilet", "sinkbasin", "countertop"],
        "actions": [
            "look",
            "go to toilet 1",
            "take soapbar 2 from toilet 1",
            "go to sinkbasin 2",
            "clean soapbar 2 with sinkbasin 2",
            "go to countertop 1",
            "move soapbar 2 to countertop 1",
        ],
    },
    # Task 10: pick_clean_then_place_in_recep-Mug-None-CoffeeMachine-10
    10: {
        "task_type": "pick_clean_then_place_in_recep",
        "object": "mug",
        "target": "coffeemachine",
        "room_signature": [
            "toaster",
            "stoveburner",
            "sinkbasin",
            "microwave",
            "fridge",
            "countertop",
        ],
        "actions": [
            "look",
            "go to toaster 1",
            "go to stoveburner 1",
            "go to stoveburner 2",
            "go to stoveburner 3",
            "go to stoveburner 4",
            "go to sinkbasin 1",
            "go to shelf 1",
            "go to shelf 2",
            "go to shelf 3",
            "go to microwave 1",
            "open microwave 1",
            "close microwave 1",
            "go to garbagecan 1",
            "go to fridge 1",
            "open fridge 1",
            "close fridge 1",
            "go to drawer 1",
            "open drawer 1",
            "close drawer 1",
            "go to drawer 2",
            "open drawer 2",
            "close drawer 2",
            "go to drawer 3",
            "open drawer 3",
            "close drawer 3",
            "go to countertop 1",
            "go to countertop 2",
            "take mug 3 from countertop 2",
            "go to sinkbasin 1",
            "clean mug 3 with sinkbasin 1",
            "go to coffeemachine 1",
            "move mug 3 to coffeemachine 1",
        ],
    },
    # Task 11: pick_cool_then_place_in_recep-Bread-None-CounterTop-10
    11: {
        "task_type": "pick_cool_then_place_in_recep",
        "object": "bread",
        "target": "countertop",
        "room_signature": ["microwave", "garbagecan", "fridge", "countertop"],
        "actions": [
            "look",
            "go to microwave 1",
            "open microwave 1",
            "close microwave 1",
            "go to garbagecan 1",
            "go to fridge 1",
            "open fridge 1",
            "close fridge 1",
            "go to countertop 1",
            "go to countertop 2",
            "go to countertop 3",
            "take bread 2 from countertop 3",
            "go to fridge 1",
            "cool bread 2 with fridge 1",
            "go to countertop 1",
            "move bread 2 to countertop 1",
        ],
    },
    # Task 12: pick_cool_then_place_in_recep-Lettuce-None-CounterTop-10
    12: {
        "task_type": "pick_cool_then_place_in_recep",
        "object": "lettuce",
        "target": "countertop",
        "room_signature": ["sinkbasin", "garbagecan", "fridge", "countertop"],
        "actions": [
            "look",
            "go to sinkbasin 1",
            "go to garbagecan 1",
            "go to fridge 1",
            "open fridge 1",
            "close fridge 1",
            "go to countertop 1",
            "take lettuce 1 from countertop 1",
            "go to fridge 1",
            "cool lettuce 1 with fridge 1",
            "go to countertop 1",
            "move lettuce 1 to countertop 1",
        ],
    },
    # Task 13: pick_cool_then_place_in_recep-Mug-None-Cabinet-10
    13: {
        "task_type": "pick_cool_then_place_in_recep",
        "object": "mug",
        "target": "cabinet",
        "room_signature": [
            "toaster",
            "stoveburner",
            "sinkbasin",
            "microwave",
            "fridge",
            "countertop",
            "cabinet",
        ],
        "actions": [
            "look",
            "go to toaster 1",
            "go to stoveburner 1",
            "go to stoveburner 2",
            "go to stoveburner 3",
            "go to stoveburner 4",
            "go to sinkbasin 1",
            "go to shelf 1",
            "go to shelf 2",
            "go to shelf 3",
            "go to microwave 1",
            "open microwave 1",
            "close microwave 1",
            "go to garbagecan 1",
            "go to fridge 1",
            "open fridge 1",
            "close fridge 1",
            "go to drawer 1",
            "open drawer 1",
            "close drawer 1",
            "go to drawer 2",
            "open drawer 2",
            "close drawer 2",
            "go to drawer 3",
            "open drawer 3",
            "close drawer 3",
            "go to countertop 1",
            "take mug 2 from countertop 1",
            "go to fridge 1",
            "cool mug 2 with fridge 1",
            "go to cabinet 1",
            "move mug 2 to cabinet 1",
        ],
    },
    # Task 14: pick_heat_then_place_in_recep-Apple-None-GarbageCan-10
    14: {
        "task_type": "pick_heat_then_place_in_recep",
        "object": "apple",
        "target": "garbagecan",
        "room_signature": ["sinkbasin", "microwave", "garbagecan"],
        "actions": [
            "look",
            "go to sinkbasin 1",
            "take apple 3 from sinkbasin 1",
            "go to microwave 1",
            "heat apple 3 with microwave 1",
            "go to garbagecan 1",
            "move apple 3 to garbagecan 1",
        ],
    },
    # Task 15: pick_heat_then_place_in_recep-Cup-None-Cabinet-10
    15: {
        "task_type": "pick_heat_then_place_in_recep",
        "object": "cup",
        "target": "cabinet",
        "room_signature": [
            "toaster",
            "stoveburner",
            "sinkbasin",
            "microwave",
            "cabinet",
        ],
        "actions": [
            "look",
            "go to toaster 1",
            "go to stoveburner 1",
            "go to stoveburner 2",
            "go to stoveburner 3",
            "go to stoveburner 4",
            "go to sinkbasin 1",
            "take cup 4 from sinkbasin 1",
            "go to microwave 1",
            "heat cup 4 with microwave 1",
            "go to cabinet 1",
            "move cup 4 to cabinet 1",
        ],
    },
    # Task 16: pick_heat_then_place_in_recep-Cup-None-Cabinet-10 (variant)
    16: {
        "task_type": "pick_heat_then_place_in_recep",
        "object": "cup",
        "target": "cabinet",
        "room_signature": [
            "toaster",
            "stoveburner",
            "sinkbasin",
            "microwave",
            "cabinet",
        ],
        "actions": [
            "look",
            "go to toaster 1",
            "go to stoveburner 1",
            "go to stoveburner 2",
            "go to stoveburner 3",
            "go to stoveburner 4",
            "go to sinkbasin 1",
            "take cup 3 from sinkbasin 1",
            "go to microwave 1",
            "heat cup 3 with microwave 1",
            "go to cabinet 1",
            "move cup 3 to cabinet 1",
        ],
    },
    # Task 17: pick_two_obj_and_place-Pillow-None-Sofa-219 (incomplete - hits step limit)
    17: {
        "task_type": "pick_two_obj_and_place",
        "object": "pillow",
        "target": "sofa",
        "room_signature": ["sofa", "armchair", "cabinet", "shelf", "drawer"],
        "actions": [
            "look",
            "go to sofa 1",
            "go to armchair 1",
            "take pillow 1 from armchair 1",
            "go to sofa 1",
            "move pillow 1 to sofa 1",
            "go to armchair 1",
            "take pillow 2 from armchair 1",
            "go to sofa 1",
            "move pillow 2 to sofa 1",
        ],
    },
    # Task 18: pick_two_obj_and_place-KeyChain-None-Safe-219 (incomplete)
    18: {
        "task_type": "pick_two_obj_and_place",
        "object": "keychain",
        "target": "safe",
        "room_signature": ["sofa", "safe", "cabinet", "dresser"],
        "actions": [
            "look",
            "go to sofa 1",
            "take keychain 4 from sofa 1",
            "go to safe 1",
            "open safe 1",
            "move keychain 4 to safe 1",
            "go to dresser 1",
            "take keychain 2 from dresser 1",
            "go to safe 1",
            "move keychain 2 to safe 1",
        ],
    },
    # Task 19: pick_two_obj_and_place-Pillow-None-Sofa-219 (variant, incomplete)
    19: {
        "task_type": "pick_two_obj_and_place",
        "object": "pillow",
        "target": "sofa",
        "room_signature": ["sofa", "armchair", "shelf", "garbagecan", "cabinet"],
        "actions": [
            "look",
            "go to sofa 1",
            "go to armchair 1",
            "take pillow 2 from armchair 1",
            "go to sofa 1",
            "move pillow 2 to sofa 1",
            "go to armchair 1",
            "take pillow 1 from armchair 1",
            "go to sofa 1",
            "move pillow 1 to sofa 1",
        ],
    },
}


def identify_task_type(goal: str) -> str:
    """Identify task type from goal text."""
    goal_lower = goal.lower()

    # Check for look_at_obj_in_light - can use "look at", "examine", etc. with lamp/light
    if ("look" in goal_lower or "examine" in goal_lower) and (
        "light" in goal_lower or "lamp" in goal_lower
    ):
        return "look_at_obj_in_light"
    elif "clean" in goal_lower:
        return "pick_clean_then_place_in_recep"
    elif "heat" in goal_lower or "hot" in goal_lower:
        return "pick_heat_then_place_in_recep"
    elif "cool" in goal_lower or "cold" in goal_lower:
        return "pick_cool_then_place_in_recep"
    elif "two" in goal_lower or "2" in goal_lower:
        return "pick_two_obj_and_place"
    else:
        return "pick_and_place_simple"


def extract_room_signature(observation: str) -> List[str]:
    """Extract room signature (list of receptacle types) from observation."""
    import re

    # Common receptacle patterns
    receptacle_pattern = (
        r"\b(bed|desk|drawer|shelf|cabinet|safe|dresser|sidetable|"
        r"sofa|armchair|toilet|sinkbasin|countertop|microwave|fridge|"
        r"toaster|stoveburner|garbagecan|coffeemachine)\s*\d*"
    )

    matches = re.findall(receptacle_pattern, observation.lower())
    return list(set(matches))


def extract_observable_objects(observation: str) -> List[str]:
    """Extract object types mentioned in the observation."""
    import re

    # Common object patterns in ALFWorld
    object_pattern = (
        r"\b(book|mug|cd|alarmclock|saltshaker|watch|vase|pan|soapbar|"
        r"bread|lettuce|apple|cup|pillow|keychain)\s*\d*"
    )

    matches = re.findall(object_pattern, observation.lower())
    return list(set(matches))


def match_task(goal: str, observation: str) -> int:
    """Match goal and observation to a task index.

    Returns the best matching task index, or -1 if no match found.
    """
    task_type = identify_task_type(goal)
    room_sig = set(extract_room_signature(observation))
    observable_objects = set(extract_observable_objects(observation))

    best_match = -1
    best_score = 0

    for task_id, task_data in EXPERT_TRAJECTORIES.items():
        if task_data["task_type"] != task_type:
            continue

        score = 0.0

        # Score based on room signature overlap (weight: 0.3)
        task_sig = set(task_data["room_signature"])
        if task_sig:
            overlap = len(room_sig & task_sig)
            total = len(room_sig | task_sig)
            score += 0.3 * (overlap / total if total > 0 else 0)

        # Score based on object match (weight: 0.7) - higher priority
        task_object = task_data.get("object", "").lower()
        if task_object:
            if task_object in observable_objects:
                score += 0.7  # Strong match if object is visible
            elif task_object in observation.lower():
                score += 0.5  # Partial match if object mentioned anywhere

        if score > best_score:
            best_score = score
            best_match = task_id

    return best_match


def get_trajectory(task_id: int) -> List[str]:
    """Get the expert action trajectory for a task."""
    if task_id in EXPERT_TRAJECTORIES:
        return EXPERT_TRAJECTORIES[task_id]["actions"]
    return []


def get_action_with_reasoning(
    task_id: int,
    step: int,
    goal: str,
    observation: str,
    profile_name: str = "expert",
    trajectory_override: List[str] = None,
) -> Tuple[str, str]:
    """Get action and reasoning for a specific step.

    Args:
        task_id: Task ID from EXPERT_TRAJECTORIES
        step: Current step number (0-indexed)
        goal: Task goal description
        observation: Current environment observation
        profile_name: Hardcoded profile name (expert, competent, novice, etc.)
        trajectory_override: Optional pre-degraded trajectory to use instead of expert

    Returns:
        Tuple of (action, reasoning). Returns ("look", fallback_reasoning) if
        step is out of bounds or task not found.
    """
    profile = get_profile(profile_name)
    if task_id not in EXPERT_TRAJECTORIES:
        fallback_reasoning = generate_reasoning(
            "look", step, goal, observation, profile
        )
        return ("look", fallback_reasoning)

    # Use override trajectory if provided, otherwise use expert trajectory
    if trajectory_override is not None:
        trajectory = trajectory_override
    else:
        trajectory = EXPERT_TRAJECTORIES[task_id]["actions"]

    if step >= len(trajectory):
        return (
            "look",
            generate_reasoning("look", step, goal, observation, profile),
        )

    action = trajectory[step]
    reasoning = generate_reasoning(action, step, goal, observation, profile)

    return (action, reasoning)


# =============================================================================
# OBSERVATION-AWARE STRATEGY EXECUTION
# =============================================================================


@dataclass
class AgentState:
    """State tracking for observation-aware agent."""

    goal: str = ""
    task_type: str = ""
    target_object: str = ""  # e.g., "saltshaker"
    target_receptacle: str = ""  # e.g., "cabinet"
    transform_type: Optional[str] = None  # "clean", "heat", "cool", or None

    # Current status
    current_location: Optional[str] = None
    inventory: List[str] = field(default_factory=list)
    visited_locations: Set[str] = field(default_factory=set)
    searched_containers: Set[str] = field(default_factory=set)

    # Task progress
    object_acquired: bool = False
    object_transformed: bool = False  # cleaned/heated/cooled
    object_placed: bool = False
    objects_placed_count: int = 0  # For pick_two tasks

    # Phase tracking
    phase: str = "search"  # search, acquire, transform, navigate, place

    # For pick_two tasks
    target_count: int = 1

    # Last action chosen by the policy (used for deterministic “overthinking” behaviors)
    last_action: Optional[str] = None


def extract_target_object(goal: str) -> str:
    """Extract the target object type from the goal.

    Examples:
        "put some saltshaker on cabinet" -> "saltshaker"
        "clean some pan and put it in countertop" -> "pan"
        "examine the cd with the desklamp" -> "cd"
    """
    goal_lower = goal.lower()

    # Common object patterns
    objects = [
        "saltshaker",
        "book",
        "mug",
        "cd",
        "alarmclock",
        "watch",
        "vase",
        "pan",
        "soapbar",
        "bread",
        "lettuce",
        "apple",
        "cup",
        "pillow",
        "keychain",
        "potato",
        "tomato",
        "egg",
        "knife",
        "fork",
        "spoon",
        "plate",
        "bowl",
        "pot",
        "spatula",
        "pencil",
        "pen",
        "cellphone",
        "laptop",
        "remotecontrol",
        "tissuebox",
        "newspaper",
        "creditcard",
        "glassbottle",
        "winebottle",
        "spraybottle",
        "cloth",
        "candle",
        "statue",
        "houseplant",
        "floorlamp",
        "television",
        "box",
    ]

    for obj in objects:
        if obj in goal_lower:
            return obj

    # Fallback: try to extract from patterns like "put some X" or "clean some X"
    patterns = [
        r"put (?:some |a |the )?(\w+)",
        r"clean (?:some |a |the )?(\w+)",
        r"heat (?:some |a |the )?(\w+)",
        r"cool (?:some |a |the )?(\w+)",
        r"examine (?:the |a )?(\w+)",
        r"look at (?:the |a )?(\w+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, goal_lower)
        if match:
            return match.group(1)

    return ""


def extract_target_receptacle(goal: str) -> str:
    """Extract the target receptacle type from the goal.

    Examples:
        "put some saltshaker on cabinet" -> "cabinet"
        "put it in countertop" -> "countertop"
    """
    goal_lower = goal.lower()

    # Common receptacle patterns
    receptacles = [
        "cabinet",
        "countertop",
        "shelf",
        "drawer",
        "safe",
        "dresser",
        "sidetable",
        "desk",
        "bed",
        "sofa",
        "armchair",
        "toilet",
        "sinkbasin",
        "microwave",
        "fridge",
        "garbagecan",
        "coffeemachine",
        "bathtub",
        "bathtubbasin",
        "cart",
        "ottoman",
        "tvstand",
    ]

    # Check for "on/in/at <receptacle>" patterns
    patterns = [
        r"(?:on|in|at|to) (?:a |the )?(\w+)(?:\s*\d*)?(?:\s*$|\.)",
        r"put .+ (?:on|in|at) (?:a |the )?(\w+)",
        r"place .+ (?:on|in|at) (?:a |the )?(\w+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, goal_lower)
        if match:
            candidate = match.group(1).lower()
            for recep in receptacles:
                if candidate.startswith(recep):
                    return recep

    # Fallback: check if any receptacle is mentioned
    for recep in receptacles:
        if recep in goal_lower:
            return recep

    return ""


def extract_transform_type(goal: str) -> Optional[str]:
    """Extract transformation type from goal (clean/heat/cool)."""
    goal_lower = goal.lower()

    if "clean" in goal_lower:
        return "clean"
    elif "heat" in goal_lower or "hot" in goal_lower:
        return "heat"
    elif "cool" in goal_lower or "cold" in goal_lower:
        return "cool"
    return None


def parse_observation(observation: str) -> Dict[str, Any]:
    """Parse observation to extract location, visible objects, and inventory.

    Returns:
        Dict with keys:
        - location: Current location name (e.g., "shelf 1") or None
        - visible_objects: List of object names visible at current location
        - inventory: List of objects being carried
        - is_empty: Whether current location shows "nothing" or "see nothing"
        - receptacle_contents: Dict mapping receptacle to list of objects
    """
    obs_lower = observation.lower()
    result = {
        "location": None,
        "visible_objects": [],
        "inventory": [],
        "is_empty": False,
        "receptacle_contents": {},
        "action_failed": False,
    }

    # Detect if at a specific location
    # Pattern: "You arrive at shelf 1" or "You are facing the cabinet 1"
    location_patterns = [
        r"you arrive at ([\w\s]+\d+)",
        r"you are facing the ([\w\s]+\d+)",
        r"on the ([\w\s]+\d+), you see",
    ]

    for pattern in location_patterns:
        match = re.search(pattern, obs_lower)
        if match:
            result["location"] = match.group(1).strip()
            break

    # Extract visible objects at current location
    # Pattern: "On the shelf 1, you see a saltshaker 1, a book 2"
    content_match = re.search(r"on the ([\w\s]+\d+), you see (.+?)(?:\.|$)", obs_lower)
    if content_match:
        location = content_match.group(1).strip()
        contents_str = content_match.group(2)
        result["location"] = location

        # Parse individual objects
        # Pattern: "a saltshaker 1" or "saltshaker 1"
        obj_pattern = r"(?:a |an |the )?([\w]+)\s*(\d+)"
        objects = re.findall(obj_pattern, contents_str)
        result["visible_objects"] = [f"{obj} {num}" for obj, num in objects]
        result["receptacle_contents"][location] = result["visible_objects"]

    # Detect empty location
    if "you see nothing" in obs_lower or "see nothing" in obs_lower:
        result["is_empty"] = True

    # Detect action failure
    if "nothing happens" in obs_lower:
        result["action_failed"] = True

    # Parse inventory
    # Pattern: "You are carrying: a saltshaker 1"
    inv_match = re.search(r"you are carrying:?\s*(.+?)(?:\.|$)", obs_lower)
    if inv_match:
        inv_str = inv_match.group(1)
        obj_pattern = r"(?:a |an |the )?([\w]+)\s*(\d+)"
        objects = re.findall(obj_pattern, inv_str)
        result["inventory"] = [f"{obj} {num}" for obj, num in objects]

    # Also check "You pick up the X" or "You put the X"
    pickup_match = re.search(r"you (?:pick up|take) the ([\w]+)\s*(\d+)", obs_lower)
    if pickup_match:
        obj = f"{pickup_match.group(1)} {pickup_match.group(2)}"
        if obj not in result["inventory"]:
            result["inventory"].append(obj)

    return result


def parse_valid_actions(valid_actions_text: str) -> List[str]:
    """Parse valid actions from the <valid_actions> tag content.

    Args:
        valid_actions_text: Text content from valid_actions tag, typically
            formatted as a list like "- go to shelf 1\n- take mug 1 from shelf 1"

    Returns:
        List of valid action strings
    """
    actions = []
    for line in valid_actions_text.strip().split("\n"):
        line = line.strip()
        # Remove list prefixes like "- " or "* " or numbers
        if line.startswith("- "):
            line = line[2:]
        elif line.startswith("* "):
            line = line[2:]
        elif re.match(r"^\d+\.\s*", line):
            line = re.sub(r"^\d+\.\s*", "", line)

        if line:
            actions.append(line.strip())

    return actions


def find_action_in_valid(
    intended_action: str, valid_actions: List[str]
) -> Optional[str]:
    """Find matching action in valid_actions list.

    Handles case-insensitive matching and partial matching.

    Returns:
        The matching action from valid_actions, or None if not found.
    """
    if not valid_actions:
        return None

    intended_lower = intended_action.lower().strip()

    # Exact match (case-insensitive)
    for action in valid_actions:
        if action.lower().strip() == intended_lower:
            return action

    # Substring match
    for action in valid_actions:
        action_lower = action.lower()
        if intended_lower in action_lower or action_lower in intended_lower:
            return action

    return None


def find_take_action(target_object: str, valid_actions: List[str]) -> Optional[str]:
    """Find a 'take' action for the target object type in valid_actions.

    Args:
        target_object: Object type to take (e.g., "saltshaker")
        valid_actions: List of valid actions

    Returns:
        The take action if found, None otherwise.
    """
    target_lower = target_object.lower()

    for action in valid_actions:
        action_lower = action.lower()
        if action_lower.startswith("take ") and target_lower in action_lower:
            return action

    return None


def find_place_action(
    target_object: str, target_receptacle: str, valid_actions: List[str]
) -> Optional[str]:
    """Find a 'move' or 'put' action for placing object on the TARGET receptacle.

    IMPORTANT: Only returns an action if it places on the correct receptacle type.
    Does NOT fall back to placing on wrong locations.

    Args:
        target_object: Object type to place (e.g., "saltshaker")
        target_receptacle: Receptacle type (e.g., "cabinet")
        valid_actions: List of valid actions

    Returns:
        The place action if found AND it targets the correct receptacle, None otherwise.
    """
    target_obj_lower = target_object.lower()
    target_recep_lower = target_receptacle.lower()

    for action in valid_actions:
        action_lower = action.lower()
        is_place = action_lower.startswith("move ") or action_lower.startswith("put ")
        if is_place and target_obj_lower in action_lower:
            # ONLY return if destination matches target receptacle type
            if target_recep_lower in action_lower:
                return action

    # Do NOT fallback to placing on wrong receptacle - return None instead
    # The caller should navigate to the correct location first
    return None


def find_transform_action(
    target_object: str, transform_type: str, valid_actions: List[str]
) -> Optional[str]:
    """Find a transform action (clean/heat/cool) for the object.

    Args:
        target_object: Object type (e.g., "pan")
        transform_type: "clean", "heat", or "cool"
        valid_actions: List of valid actions

    Returns:
        The transform action if found, None otherwise.
    """
    target_lower = target_object.lower()

    for action in valid_actions:
        action_lower = action.lower()
        if (
            action_lower.startswith(f"{transform_type} ")
            and target_lower in action_lower
        ):
            return action

    return None


def find_use_action(target: str, valid_actions: List[str]) -> Optional[str]:
    """Find a 'use' action for the target (e.g., desklamp).

    Args:
        target: Target to use (e.g., "desklamp")
        valid_actions: List of valid actions

    Returns:
        The use action if found, None otherwise.
    """
    target_lower = target.lower()

    for action in valid_actions:
        action_lower = action.lower()
        if action_lower.startswith("use ") and target_lower in action_lower:
            return action

    return None


def find_goto_action(
    receptacle_type: str, valid_actions: List[str], visited: Set[str]
) -> Optional[str]:
    """Find a 'go to' action for an unvisited location of the given type.

    Args:
        receptacle_type: Type of receptacle (e.g., "shelf", "cabinet")
        valid_actions: List of valid actions
        visited: Set of already-visited locations

    Returns:
        The go to action if found, None otherwise.
    """
    recep_lower = receptacle_type.lower()
    visited_lower = {v.lower() for v in visited}

    for action in valid_actions:
        action_lower = action.lower()
        if action_lower.startswith("go to ") and recep_lower in action_lower:
            # Extract the destination
            dest = action_lower[6:].strip()
            if dest not in visited_lower:
                return action

    return None


def find_open_action(receptacle: str, valid_actions: List[str]) -> Optional[str]:
    """Find an 'open' action for the given receptacle."""
    recep_lower = receptacle.lower()

    for action in valid_actions:
        action_lower = action.lower()
        if action_lower.startswith("open ") and recep_lower in action_lower:
            return action

    return None


def get_search_priority(task_type: str, target_object: str) -> List[str]:
    """Get prioritized list of receptacle types to search for the target object.

    Priorities are derived from ground truth statistics in the ALFRED dataset
    (traj_data.json files), representing actual object placements.

    Args:
        task_type: The task type
        target_object: The object to find

    Returns:
        List of receptacle types in priority order
    """
    # Ground truth-based object locations (derived from ALFRED traj_data.json analysis)
    # Each list is ordered by frequency of occurrence in the dataset
    object_locations = {
        # Kitchen items
        "apple": ["countertop", "sinkbasin", "diningtable", "fridge", "microwave"],
        "bread": ["countertop", "diningtable", "fridge", "microwave"],
        "butterknife": ["countertop", "drawer", "diningtable"],
        "cup": ["cabinet", "diningtable", "sinkbasin", "microwave", "countertop"],
        "egg": ["sinkbasin", "countertop", "cabinet", "fridge", "diningtable"],
        "fork": ["countertop", "sinkbasin", "diningtable", "drawer"],
        "knife": ["countertop", "diningtable", "drawer"],
        "ladle": ["countertop", "diningtable", "drawer"],
        "lettuce": ["countertop", "diningtable", "fridge", "sinkbasin"],
        "mug": ["cabinet", "diningtable", "coffeemachine", "shelf", "countertop"],
        "pan": ["stoveburner", "countertop", "cabinet", "sinkbasin"],
        "plate": ["countertop", "diningtable", "shelf", "cabinet"],
        "potato": ["countertop", "sinkbasin", "fridge", "microwave"],
        "saltshaker": ["shelf", "cabinet", "countertop", "diningtable"],
        "spatula": ["countertop", "diningtable", "drawer"],
        "tomato": ["countertop", "diningtable", "fridge", "sinkbasin", "garbagecan"],
        "bowl": ["countertop", "desk", "shelf", "diningtable", "cabinet"],
        # Bedroom items
        "alarmclock": ["desk", "dresser", "sidetable", "shelf", "diningtable"],
        "book": ["bed", "sidetable", "desk", "shelf"],  # 60% bed, 40% sidetable
        "cd": ["desk", "shelf", "drawer", "safe", "garbagecan"],
        "cellphone": ["dresser", "bed", "shelf", "sidetable", "desk"],
        "creditcard": ["sofa", "sidetable", "diningtable", "dresser", "desk"],
        "keychain": ["dresser", "armchair", "sidetable", "desk", "sofa"],
        "pencil": ["desk", "dresser", "shelf", "drawer"],
        "pillow": ["bed", "armchair", "sofa"],  # 67% bed, 33% armchair
        "remotecontrol": ["coffeetable", "armchair", "sidetable", "sofa", "bed"],
        "vase": ["shelf", "cabinet", "dresser", "sidetable", "countertop"],
        "watch": ["diningtable", "drawer", "dresser", "sidetable", "safe"],
        # Bathroom items
        "cloth": ["cabinet", "bathtubbasin", "countertop", "drawer"],
        "handtowel": ["handtowelholder", "countertop", "cabinet"],
        "soapbar": ["countertop", "toilet", "sinkbasin", "bathtubbasin", "cabinet"],
        "soapbottle": ["countertop", "cabinet", "dresser", "toilet", "sidetable"],
        "spraybottle": ["countertop", "toilet", "sidetable", "cabinet"],
        "tissuebox": ["cabinet", "countertop", "coffeetable", "dresser"],
        "toiletpaper": ["toilet", "toiletpaperhanger", "garbagecan", "cabinet"],
        # Misc items
        "newspaper": ["sofa", "coffeetable", "sidetable", "armchair", "bed"],
        "statue": ["dresser", "shelf", "sidetable", "desk"],
    }

    if target_object.lower() in object_locations:
        return object_locations[target_object.lower()]

    # Default search order (also ground truth-informed: most common locations first)
    return [
        "countertop",  # Most common location for kitchen items
        "desk",  # Common for bedroom items
        "shelf",
        "sidetable",
        "dresser",
        "diningtable",
        "bed",  # Important for books, pillows
        "cabinet",
        "drawer",
        "fridge",
        "microwave",
        "coffeetable",
        "sofa",
        "armchair",
        "stoveburner",
        "sinkbasin",
        "toilet",
        "safe",
        "garbagecan",
    ]


def select_adaptive_action(
    state: AgentState,
    observation: str,
    valid_actions: List[str],
    step: int,
    profile: Dict[str, Any],
) -> Tuple[str, str]:
    """Select next action based on current state and observation.

    This is the main strategy execution function that adapts to the
    environment state rather than blindly following a pre-recorded trajectory.

    Args:
        state: Current agent state
        observation: Current observation from environment
        valid_actions: List of valid actions from environment
        step: 0-indexed step number for deterministic scheduling
        profile: Deterministic profile policy dict (see AGENT_PROFILES)

    Returns:
        Tuple of (action, phase_update_hint)
    """

    def _scheduled(every_n: int) -> bool:
        return bool(every_n) and every_n > 0 and (step % every_n == 0)

    rng = random.Random(step)

    # Parse the current observation
    obs_data = parse_observation(observation)

    # Update state from observation
    if obs_data["location"]:
        state.current_location = obs_data["location"]
        state.visited_locations.add(obs_data["location"].lower())

    if obs_data["inventory"]:
        state.inventory = obs_data["inventory"]
        # Check if we acquired the target object
        for item in state.inventory:
            if state.target_object.lower() in item.lower():
                state.object_acquired = True
                break

    # Check visible objects for target
    target_visible = False
    for obj in obs_data["visible_objects"]:
        if state.target_object.lower() in obj.lower():
            target_visible = True
            break

    # -------------------------------------------------------------------------
    # Deterministic profile-driven action injections (before phase logic)
    # -------------------------------------------------------------------------
    # 1) Optional “forgetfulness” to force revisits
    forget_every = int(profile.get("forget_visited_every_n_steps", 0) or 0)
    if _scheduled(forget_every) and state.visited_locations:
        keep: Set[str] = set()
        if state.current_location:
            keep.add(state.current_location.lower())
        state.visited_locations = keep

    # 2) “Overthinking”: after navigation, add an extra look
    if profile.get("extra_look_after_goto") and state.last_action:
        if state.last_action.lower().startswith("go to ") and "look" in [
            a.lower() for a in valid_actions
        ]:
            state.last_action = "look"
            return "look", "overthink_look"

    # 3) Periodic inventory checks
    inv_every = int(profile.get("inventory_check_every_n_steps", 0) or 0)
    if _scheduled(inv_every):
        for a in valid_actions:
            if a.lower().strip() == "inventory":
                state.last_action = a
                return a, "inventory_check"

    # 4) Periodic extra open/close behavior
    openclose_every = int(profile.get("extra_open_close_every_n_steps", 0) or 0)
    if _scheduled(openclose_every):
        open_actions = sorted(
            [a for a in valid_actions if a.lower().startswith("open ")],
            key=lambda s: s.lower(),
        )
        for a in open_actions:
            loc = a[5:].strip().lower()
            if loc and loc not in state.searched_containers:
                state.searched_containers.add(loc)
                state.last_action = a
                return a, "overthink_open"
        close_actions = sorted(
            [a for a in valid_actions if a.lower().startswith("close ")],
            key=lambda s: s.lower(),
        )
        if close_actions:
            state.last_action = close_actions[0]
            return close_actions[0], "overthink_close"

    # 5) Intentional stalling via look (useful for failure-prone profiles)
    stall_every = int(profile.get("stall_look_every_n_steps", 0) or 0)
    if _scheduled(stall_every):
        for a in valid_actions:
            if a.lower().strip() == "look":
                state.last_action = a
                return a, "stall"

    # 6) Premature navigation to target receptacle before acquiring object
    premature_every = int(profile.get("premature_navigate_every_n_steps", 0) or 0)
    if (
        (not state.object_acquired)
        and _scheduled(premature_every)
        and state.target_receptacle
    ):
        goto_action = find_goto_action(state.target_receptacle, valid_actions, set())
        if goto_action:
            state.last_action = goto_action
            return goto_action, "premature_navigate"

    # 7) Deterministic detours to waste steps / create extra locations
    detour_every = int(profile.get("detour_every_n_steps", 0) or 0)
    detour_types = list(profile.get("detour_location_type_priority") or [])
    if detour_types and _scheduled(detour_every):
        goto_actions = [a for a in valid_actions if a.lower().startswith("go to ")]
        # Prefer unvisited detours first, but always fall back deterministically.
        unvisited: List[str] = []
        visited: List[str] = []
        for a in goto_actions:
            dest = a[6:].strip()
            dest_l = dest.lower()
            if any(dest_l.startswith(t.lower()) for t in detour_types):
                if dest_l in state.visited_locations:
                    visited.append(a)
                else:
                    unvisited.append(a)
        candidates = unvisited or visited
        if candidates:
            chosen = sorted(candidates, key=lambda s: s.lower())[0]
            state.last_action = chosen
            return chosen, "detour"

    # Determine action based on phase and task type
    task_type = state.task_type

    # Phase: SEARCH - Find the target object
    if not state.object_acquired:
        # First check if we can take the object from current location
        if target_visible:
            take_action = find_take_action(state.target_object, valid_actions)
            if take_action:
                state.phase = "acquire"
                return take_action, "acquire"

        # Object not here or can't take it - continue searching
        search_priority = get_search_priority(task_type, state.target_object)

        # Deterministic search noise
        noise_mode = str(profile.get("search_priority_noise_mode", "none"))
        if noise_mode == "shuffle":
            rng.shuffle(search_priority)
        elif noise_mode == "swap_first" and len(search_priority) > 1:
            offset = (step * 3 + 1) % len(search_priority)
            search_priority = search_priority[offset:] + search_priority[:offset]

        # Find an unvisited location to search
        for recep_type in search_priority:
            goto_action = find_goto_action(
                recep_type, valid_actions, state.visited_locations
            )
            if goto_action:
                state.phase = "search"
                state.last_action = goto_action
                return goto_action, "search"

        # Check if we need to open containers
        openable = ["drawer", "safe", "fridge", "microwave", "cabinet"]
        for recep_type in openable:
            for action in valid_actions:
                if action.lower().startswith(f"open {recep_type}"):
                    loc = action[5:].strip()
                    if loc.lower() not in state.searched_containers:
                        state.searched_containers.add(loc.lower())
                        state.last_action = action
                        return action, "search"

        # Last resort: try any unvisited "go to" action
        for action in valid_actions:
            if action.lower().startswith("go to "):
                dest = action[6:].strip().lower()
                if dest not in state.visited_locations:
                    state.last_action = action
                    return action, "search"

        # Truly stuck - just look
        if "look" in valid_actions:
            state.last_action = "look"
            return "look", "search"
        if valid_actions:
            state.last_action = valid_actions[0]
        return valid_actions[0] if valid_actions else "look", "search"

    # Phase: TRANSFORM - Clean/heat/cool the object if needed
    if state.transform_type and not state.object_transformed:
        # Check if transform action is available
        transform_action = find_transform_action(
            state.target_object, state.transform_type, valid_actions
        )
        if transform_action:
            state.object_transformed = True
            state.phase = "navigate"
            state.last_action = transform_action
            return transform_action, "transform"

        # Need to go to transform location
        transform_locations = {
            "clean": ["sinkbasin"],
            "heat": ["microwave", "stoveburner"],
            "cool": ["fridge"],
        }
        for loc_type in transform_locations.get(state.transform_type, []):
            goto_action = find_goto_action(loc_type, valid_actions, set())
            if goto_action:
                state.phase = "transform"
                state.last_action = goto_action
                return goto_action, "transform"

    # Phase: NAVIGATE - Go to target receptacle (or use lamp for look_at tasks)
    if state.object_acquired and (not state.transform_type or state.object_transformed):
        # SPECIAL CASE: look_at_obj_in_light tasks
        # For these tasks, we need to USE the desklamp while HOLDING the object
        # We should NOT place the object first!
        if task_type == "look_at_obj_in_light":
            use_action = find_use_action("desklamp", valid_actions)
            if use_action:
                state.phase = "complete"
                state.last_action = use_action
                return use_action, "use"
            # Go to desk to find and use lamp
            goto_action = find_goto_action(
                "desk", valid_actions, state.visited_locations
            )
            if goto_action:
                state.last_action = goto_action
                return goto_action, "navigate"
            # If all desks visited, try any desk
            for action in valid_actions:
                if action.lower().startswith("go to desk"):
                    state.last_action = action
                    return action, "navigate"
            # Last resort for look_at tasks
            if "look" in valid_actions:
                state.last_action = "look"
                return "look", "fallback"
            if valid_actions:
                state.last_action = valid_actions[0]
            return valid_actions[0] if valid_actions else "look", "fallback"

        # For all other tasks: place the object on the target receptacle
        place_action = find_place_action(
            state.target_object, state.target_receptacle, valid_actions
        )
        if place_action:
            state.objects_placed_count += 1

            # Check if we've placed enough objects (for pick_two tasks)
            if state.objects_placed_count >= state.target_count:
                state.object_placed = True
                state.phase = "complete"
                state.last_action = place_action
                return place_action, "place"
            else:
                # Need to find and place more objects (pick_two task)
                # Reset search state for next object
                state.object_acquired = False
                state.object_transformed = False
                state.phase = "search"
                # Clear visited locations to allow revisiting for second object
                state.visited_locations.clear()
                state.last_action = place_action
                return place_action, "place_continue"

        # Need to navigate to target receptacle
        # For safe, may need to open it first
        if state.target_receptacle.lower() == "safe":
            open_action = find_open_action("safe", valid_actions)
            if open_action:
                state.last_action = open_action
                return open_action, "navigate"

        goto_action = find_goto_action(state.target_receptacle, valid_actions, set())
        if goto_action:
            state.phase = "navigate"
            state.last_action = goto_action
            return goto_action, "navigate"

        # Try any receptacle of the target type
        for action in valid_actions:
            if (
                action.lower().startswith("go to ")
                and state.target_receptacle.lower() in action.lower()
            ):
                state.last_action = action
                return action, "navigate"

    # Fallback - find a valid navigation action
    # Filter valid_actions to only include actual game commands
    game_actions = [
        a
        for a in valid_actions
        if a.lower().startswith(
            (
                "go to ",
                "take ",
                "move ",
                "put ",
                "open ",
                "close ",
                "use ",
                "examine ",
                "clean ",
                "heat ",
                "cool ",
            )
        )
        or a.lower() in ("look", "inventory")
    ]

    if "look" in game_actions:
        state.last_action = "look"
        return "look", "fallback"
    if game_actions:
        state.last_action = game_actions[0]
        return game_actions[0], "fallback"
    state.last_action = "look"
    return "look", "fallback"


def initialize_state_from_goal(goal: str) -> AgentState:
    """Initialize agent state from goal description.

    Args:
        goal: Task goal description

    Returns:
        Initialized AgentState
    """
    task_type = identify_task_type(goal)
    target_object = extract_target_object(goal)
    target_receptacle = extract_target_receptacle(goal)
    transform_type = extract_transform_type(goal)

    # Determine target count for pick_two tasks
    target_count = 2 if task_type == "pick_two_obj_and_place" else 1

    return AgentState(
        goal=goal,
        task_type=task_type,
        target_object=target_object,
        target_receptacle=target_receptacle,
        transform_type=transform_type,
        target_count=target_count,
    )
