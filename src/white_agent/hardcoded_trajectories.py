"""Pre-recorded expert trajectories for ALFWorld tasks.

Each trajectory contains:
- task_type: The ALFWorld task type
- object: The target object being manipulated
- target: The destination/tool
- actions: List of expert actions
- reasoning_templates: Templates for generating contextual reasoning
"""

import random
from typing import Dict, List, Any, Tuple


# Agent profiles combining reasoning and strategy quality levels
AGENT_PROFILES: Dict[str, Dict[str, str]] = {
    "expert": {
        "reasoning": "high",
        "strategy": "optimal",
        "description": "Expert agent with perfect reasoning and strategy",
    },
    "competent": {
        "reasoning": "medium",
        "strategy": "suboptimal",
        "description": "Good agent with some inefficiencies",
    },
    "novice": {
        "reasoning": "low",
        "strategy": "poor",
        "description": "Beginner agent with weak reasoning and strategy",
    },
    "lucky_guesser": {
        "reasoning": "low",
        "strategy": "optimal",
        "description": "Gets right actions but can't explain why",
    },
    "overthinker": {
        "reasoning": "high",
        "strategy": "suboptimal",
        "description": "Good reasoning but inefficient execution",
    },
}


def _generate_low_quality_reasoning() -> str:
    """Generate minimal/generic reasoning."""
    templates = [
        "Proceeding.",
        "Next step.",
        "Continuing.",
        "Moving on.",
        "",  # Sometimes no reasoning
    ]
    return random.choice(templates)


def _generate_medium_quality_reasoning(action: str) -> str:
    """Generate action-aware but generic reasoning."""
    action_lower = action.lower()

    if action_lower.startswith("go to "):
        return random.choice(
            [
                "Moving to check another location.",
                "Going to a new area.",
                "Navigating to the next spot.",
            ]
        )
    elif action_lower.startswith("take "):
        return random.choice(
            [
                "Picking up an item.",
                "Grabbing something.",
                "Taking this object.",
            ]
        )
    elif action_lower.startswith("move ") or action_lower.startswith("put "):
        return random.choice(
            [
                "Placing the item.",
                "Putting it down.",
                "Moving the object.",
            ]
        )
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
    quality: str = "high",
) -> str:
    """Generate reasoning at the specified quality level.

    Args:
        action: The action being taken
        step: Current step number
        goal: Task goal description
        observation: Current environment observation
        quality: Reasoning quality level ("high", "medium", "low")

    Returns:
        Reasoning string appropriate to the quality level
    """
    if quality == "low":
        return _generate_low_quality_reasoning()
    elif quality == "medium":
        return _generate_medium_quality_reasoning(action)
    else:  # "high" is default
        return _generate_high_quality_reasoning(action, step, goal, observation)


# --- Strategy degradation functions ---


def _add_minor_inefficiencies(actions: List[str], seed: int = None) -> List[str]:
    """Insert occasional unnecessary actions (suboptimal strategy).

    Adds ~30% chance of extra 'look' after navigation.
    """
    if seed is not None:
        random.seed(seed)

    result = []
    for action in actions:
        result.append(action)
        # 30% chance to insert unnecessary "look" after navigation
        if action.lower().startswith("go to ") and random.random() < 0.3:
            result.append("look")
    return result


def _add_major_inefficiencies(actions: List[str], seed: int = None) -> List[str]:
    """Insert random detours and unnecessary exploration (poor strategy).

    Adds ~40% chance of random detours before important actions.
    """
    if seed is not None:
        random.seed(seed)

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
    for i, action in enumerate(actions):
        # 40% chance to insert a random detour before each action
        if random.random() < 0.4:
            detour = random.choice(detour_locations)
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


def get_profile(profile_name: str) -> Dict[str, str]:
    """Get profile configuration by name.

    Args:
        profile_name: Name of the profile (expert, competent, novice, etc.)

    Returns:
        Profile dictionary with reasoning and strategy quality levels
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

    if "look" in goal_lower and ("light" in goal_lower or "lamp" in goal_lower):
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
    reasoning_quality: str = "high",
    trajectory_override: List[str] = None,
) -> Tuple[str, str]:
    """Get action and reasoning for a specific step.

    Args:
        task_id: Task ID from EXPERT_TRAJECTORIES
        step: Current step number (0-indexed)
        goal: Task goal description
        observation: Current environment observation
        reasoning_quality: Quality level for reasoning ("high", "medium", "low")
        trajectory_override: Optional pre-degraded trajectory to use instead of expert

    Returns:
        Tuple of (action, reasoning). Returns ("look", fallback_reasoning) if
        step is out of bounds or task not found.
    """
    if task_id not in EXPERT_TRAJECTORIES:
        fallback_reasoning = generate_reasoning(
            "look", step, goal, observation, reasoning_quality
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
            generate_reasoning("look", step, goal, observation, reasoning_quality),
        )

    action = trajectory[step]
    reasoning = generate_reasoning(action, step, goal, observation, reasoning_quality)

    return (action, reasoning)
