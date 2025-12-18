"""Ground truth baseline trajectory loader.

This module extracts optimal trajectories from traj_data.json files in the
ALFRED/ALFWorld dataset. These trajectories represent actual human demonstrations
and serve as the ideal baseline for evaluation.

The high-level plan from traj_data.json is converted to TextWorld commands.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)


@dataclass
class GroundTruthStep:
    """A single step from the ground truth trajectory."""

    step_num: int
    high_level_action: str  # e.g., "GotoLocation"
    args: List[str]  # e.g., ["dresser"]
    textworld_command: str  # e.g., "go to dresser 1"


@dataclass
class GroundTruthTrajectory:
    """Optimal trajectory from traj_data.json."""

    task_id: str
    task_type: str
    pddl_params: Dict[str, Any]
    steps: List[GroundTruthStep]
    raw_high_pddl: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def actions(self) -> List[str]:
        """Get list of TextWorld commands."""
        return [s.textworld_command for s in self.steps]

    @property
    def step_count(self) -> int:
        """Get total number of steps."""
        return len(self.steps)

    @property
    def success(self) -> bool:
        """Ground truth trajectories are always successful by definition."""
        return True


class ObjectIDResolver:
    """Resolves object class names to numbered TextWorld IDs.

    Parses the initial observation to build a mapping from class names
    (e.g., "dresser") to numbered IDs (e.g., "dresser 1", "dresser 2").
    """

    def __init__(self, initial_observation: str):
        """Initialize resolver from initial observation text.

        Args:
            initial_observation: The welcome text from TextWorld containing
                receptacle listings like "you see a bed 1, a desk 2, ..."
        """
        self._class_to_ids: Dict[str, List[str]] = {}
        self._parse_observation(initial_observation)

    def _parse_observation(self, observation: str) -> None:
        """Parse observation to extract object IDs."""
        # Pattern matches things like "a bed 1", "a desk 2", "a drawer 6"
        # Handles both "a" and "an" articles
        pattern = r"\b(?:a|an)\s+([a-z]+(?:\s+[a-z]+)?)\s+(\d+)"

        for match in re.finditer(pattern, observation.lower()):
            obj_class = match.group(1).strip()
            obj_num = match.group(2)
            obj_id = f"{obj_class} {obj_num}"

            if obj_class not in self._class_to_ids:
                self._class_to_ids[obj_class] = []
            if obj_id not in self._class_to_ids[obj_class]:
                self._class_to_ids[obj_class].append(obj_id)

        LOGGER.debug(
            f"Parsed {len(self._class_to_ids)} object classes from observation"
        )

    def resolve(self, class_name: str, index: int = 0) -> Optional[str]:
        """Resolve a class name to a specific ID.

        Args:
            class_name: Object class (e.g., "dresser", "desk")
            index: Which instance to return (0 = first, 1 = second, etc.)

        Returns:
            The numbered ID (e.g., "dresser 1") or None if not found.
        """
        class_name_lower = class_name.lower()

        # Handle special case mappings FIRST (before partial matching)
        # This prevents "desklamp" from matching "desk"
        special_mappings = {
            "sinkbasin": ["sinkbasin", "sink basin", "sink"],
            "bathtubbasin": ["bathtubbasin", "bathtub basin", "bathtub"],
            "sidetable": ["sidetable", "side table"],
            "coffeetable": ["coffeetable", "coffee table"],
            "diningtable": ["diningtable", "dining table"],
            "countertop": ["countertop", "counter top", "counter"],
            "garbagecan": ["garbagecan", "garbage can"],
            "desklamp": ["desklamp", "desk lamp"],
            "floorlamp": ["floorlamp", "floor lamp"],
        }

        if class_name_lower in special_mappings:
            for variant in special_mappings[class_name_lower]:
                if variant in self._class_to_ids:
                    ids = self._class_to_ids[variant]
                    if index < len(ids):
                        return ids[index]
                    return ids[0] if ids else None
            # Object not found in observation - return None (caller will use default)
            LOGGER.debug(
                f"Object class '{class_name}' not found in observation (may be hidden)"
            )
            return None

        # Direct match
        if class_name_lower in self._class_to_ids:
            ids = self._class_to_ids[class_name_lower]
            if index < len(ids):
                return ids[index]
            return ids[0] if ids else None

        # Try partial match ONLY for non-special cases
        # Exclude matches where one name is a substring of another but they're different objects
        # e.g., don't match "desklamp" to "desk"
        for stored_class, ids in self._class_to_ids.items():
            # Only match if names are very similar (not just substring)
            if class_name_lower == stored_class or stored_class == class_name_lower:
                if index < len(ids):
                    return ids[index]
                return ids[0] if ids else None

        LOGGER.debug(
            f"Could not resolve object class '{class_name}' (may not be visible)"
        )
        return None

    def get_first(self, class_name: str) -> Optional[str]:
        """Get the first instance of a class."""
        return self.resolve(class_name, 0)

    def get_all(self, class_name: str) -> List[str]:
        """Get all instances of a class."""
        class_name_lower = class_name.lower()
        if class_name_lower in self._class_to_ids:
            return list(self._class_to_ids[class_name_lower])
        return []


class GroundTruthConverter:
    """Converts high-level PDDL actions to TextWorld commands."""

    def __init__(
        self,
        resolver: ObjectIDResolver,
        pddl_params: Dict[str, Any],
    ):
        """Initialize converter.

        Args:
            resolver: Object ID resolver for the current scene
            pddl_params: Task parameters (object_target, parent_target, etc.)
        """
        self._resolver = resolver
        self._pddl_params = pddl_params
        self._current_receptacle: Optional[str] = None
        self._inventory_object: Optional[str] = None
        self._object_instance_count: Dict[str, int] = {}

    def convert(self, high_pddl: List[Dict[str, Any]]) -> List[GroundTruthStep]:
        """Convert high-level PDDL plan to TextWorld commands.

        Args:
            high_pddl: List of high-level actions from traj_data.json

        Returns:
            List of GroundTruthStep with TextWorld commands
        """
        steps = []
        step_num = 0

        for action_data in high_pddl:
            discrete = action_data.get("discrete_action", {})
            action_type = discrete.get("action", "")
            args = discrete.get("args", [])

            # Skip NoOp/End actions
            if action_type in ("NoOp", "End", ""):
                continue

            tw_command = self._convert_action(action_type, args)
            if tw_command:
                step_num += 1
                steps.append(
                    GroundTruthStep(
                        step_num=step_num,
                        high_level_action=action_type,
                        args=args,
                        textworld_command=tw_command,
                    )
                )

        return steps

    def _convert_action(self, action_type: str, args: List[str]) -> Optional[str]:
        """Convert a single high-level action to TextWorld command."""

        if action_type == "GotoLocation":
            return self._convert_goto(args)

        elif action_type == "PickupObject":
            return self._convert_pickup(args)

        elif action_type == "PutObject":
            return self._convert_put(args)

        elif action_type == "ToggleObject":
            return self._convert_toggle(args)

        elif action_type == "OpenObject":
            return self._convert_open(args)

        elif action_type == "CloseObject":
            return self._convert_close(args)

        elif action_type == "HeatObject":
            return self._convert_heat(args)

        elif action_type == "CoolObject":
            return self._convert_cool(args)

        elif action_type == "CleanObject":
            return self._convert_clean(args)

        elif action_type == "SliceObject":
            return self._convert_slice(args)

        else:
            LOGGER.warning(f"Unknown action type: {action_type}")
            return None

    def _convert_goto(self, args: List[str]) -> Optional[str]:
        """Convert GotoLocation to 'go to <receptacle>'.

        Note: In TextWorld/ALFWorld, you can only navigate to receptacles (bed, desk,
        shelf, etc.), NOT to small objects (desklamp, microwave contents, etc.).
        Objects like desklamps are ON receptacles, so we map to the parent receptacle.
        """
        if not args:
            return None

        target_class = args[0].lower()

        # Map non-navigatable objects to their parent receptacles
        # These objects exist ON receptacles and cannot be navigated to directly
        object_to_receptacle_map = {
            "desklamp": "desk",
            "floorlamp": "floorlamp",  # Floorlamp is actually navigatable
            "laptopcomputer": "desk",
            "laptop": "desk",
            "alarmclock": "desk",  # Often on desk/sidetable
            "book": "bed",  # For look_at tasks, book is often on bed
            "coffeemachine": "countertop",
            "toaster": "countertop",
            "stoveknob": "stoveburner",
        }

        # If target is a non-navigatable object, find its parent receptacle
        if target_class in object_to_receptacle_map:
            parent_class = object_to_receptacle_map[target_class]
            target_id = self._resolver.get_first(parent_class)
            if target_id:
                LOGGER.debug(
                    f"Mapped non-navigatable '{target_class}' to parent receptacle '{target_id}'"
                )
                self._current_receptacle = target_id
                return f"go to {target_id}"

        # Standard resolution for navigatable receptacles
        target_id = self._resolver.get_first(target_class)

        if not target_id:
            # Fallback: try with "1" suffix
            target_id = f"{target_class.lower()} 1"

        self._current_receptacle = target_id
        return f"go to {target_id}"

    def _convert_pickup(self, args: List[str]) -> Optional[str]:
        """Convert PickupObject to 'take <object> from <receptacle>'."""
        if not args:
            return None

        obj_class = args[0]

        # Track which instance we're picking up
        if obj_class not in self._object_instance_count:
            self._object_instance_count[obj_class] = 0

        instance_idx = self._object_instance_count[obj_class]
        obj_id = self._resolver.resolve(obj_class, instance_idx)

        if not obj_id:
            obj_id = f"{obj_class.lower()} 1"

        self._inventory_object = obj_id
        self._object_instance_count[obj_class] += 1

        if self._current_receptacle:
            return f"take {obj_id} from {self._current_receptacle}"
        else:
            return f"take {obj_id}"

    def _convert_put(self, args: List[str]) -> Optional[str]:
        """Convert PutObject to 'put <object> in/on <receptacle>'."""
        if len(args) < 2:
            return None

        obj_class = args[0]
        recep_class = args[1]

        # Use inventory object if available
        obj_id = self._inventory_object or self._resolver.get_first(obj_class)
        if not obj_id:
            obj_id = f"{obj_class.lower()} 1"

        recep_id = self._resolver.get_first(recep_class)
        if not recep_id:
            recep_id = f"{recep_class.lower()} 1"

        self._inventory_object = None
        return f"put {obj_id} in/on {recep_id}"

    def _convert_toggle(self, args: List[str]) -> Optional[str]:
        """Convert ToggleObject to 'use <object>'."""
        if not args:
            return None

        obj_class = args[0]
        obj_id = self._resolver.get_first(obj_class)

        if not obj_id:
            obj_id = f"{obj_class.lower()} 1"

        return f"use {obj_id}"

    def _convert_open(self, args: List[str]) -> Optional[str]:
        """Convert OpenObject to 'open <receptacle>'."""
        if not args:
            return None

        recep_class = args[0]
        recep_id = self._resolver.get_first(recep_class)

        if not recep_id:
            recep_id = f"{recep_class.lower()} 1"

        return f"open {recep_id}"

    def _convert_close(self, args: List[str]) -> Optional[str]:
        """Convert CloseObject to 'close <receptacle>'."""
        if not args:
            return None

        recep_class = args[0]
        recep_id = self._resolver.get_first(recep_class)

        if not recep_id:
            recep_id = f"{recep_class.lower()} 1"

        return f"close {recep_id}"

    def _convert_heat(self, args: List[str]) -> Optional[str]:
        """Convert HeatObject to 'heat <object> with microwave 1'."""
        if not args:
            return None

        obj_class = args[0]
        obj_id = self._inventory_object or self._resolver.get_first(obj_class)

        if not obj_id:
            obj_id = f"{obj_class.lower()} 1"

        microwave_id = self._resolver.get_first("microwave") or "microwave 1"
        return f"heat {obj_id} with {microwave_id}"

    def _convert_cool(self, args: List[str]) -> Optional[str]:
        """Convert CoolObject to 'cool <object> with fridge 1'."""
        if not args:
            return None

        obj_class = args[0]
        obj_id = self._inventory_object or self._resolver.get_first(obj_class)

        if not obj_id:
            obj_id = f"{obj_class.lower()} 1"

        fridge_id = self._resolver.get_first("fridge") or "fridge 1"
        return f"cool {obj_id} with {fridge_id}"

    def _convert_clean(self, args: List[str]) -> Optional[str]:
        """Convert CleanObject to 'clean <object> with sinkbasin 1'."""
        if not args:
            return None

        obj_class = args[0]
        obj_id = self._inventory_object or self._resolver.get_first(obj_class)

        if not obj_id:
            obj_id = f"{obj_class.lower()} 1"

        # Try sinkbasin first, then bathtubbasin
        sink_id = (
            self._resolver.get_first("sinkbasin")
            or self._resolver.get_first("bathtubbasin")
            or "sinkbasin 1"
        )
        return f"clean {obj_id} with {sink_id}"

    def _convert_slice(self, args: List[str]) -> Optional[str]:
        """Convert SliceObject to 'slice <object> with knife 1'."""
        if not args:
            return None

        obj_class = args[0]
        obj_id = self._resolver.get_first(obj_class)

        if not obj_id:
            obj_id = f"{obj_class.lower()} 1"

        knife_id = self._resolver.get_first("knife") or "knife 1"
        return f"slice {obj_id} with {knife_id}"


def load_ground_truth_trajectory(
    game_file_path: str,
    split: str,
    initial_observation: str,
    alfworld_data: Optional[str] = None,
) -> Optional[GroundTruthTrajectory]:
    """Load and convert ground truth trajectory from traj_data.json.

    Args:
        game_file_path: Relative path like "look_at_obj_in_light-.../trial_..."
        split: Dataset split ("train", "valid_seen", "valid_unseen")
        initial_observation: Initial observation text for ID resolution
        alfworld_data: Path to ALFWORLD_DATA (defaults to env var or ~/.cache/alfworld)

    Returns:
        GroundTruthTrajectory or None if loading fails
    """
    if alfworld_data is None:
        alfworld_data = os.environ.get(
            "ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld")
        )

    # Build path to traj_data.json
    traj_data_path = (
        Path(alfworld_data) / "json_2.1.1" / split / game_file_path / "traj_data.json"
    )

    if not traj_data_path.exists():
        LOGGER.warning(f"Ground truth file not found: {traj_data_path}")
        return None

    try:
        with open(traj_data_path, "r") as f:
            traj_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        LOGGER.error(f"Failed to load ground truth: {e}")
        return None

    # Extract required fields
    task_id = traj_data.get("task_id", game_file_path)
    task_type = traj_data.get("task_type", "unknown")
    pddl_params = traj_data.get("pddl_params", {})
    high_pddl = traj_data.get("plan", {}).get("high_pddl", [])

    if not high_pddl:
        LOGGER.warning(f"No high_pddl plan found in {traj_data_path}")
        return None

    # Create resolver and converter
    resolver = ObjectIDResolver(initial_observation)
    converter = GroundTruthConverter(resolver, pddl_params)

    # Convert to TextWorld commands
    steps = converter.convert(high_pddl)

    if not steps:
        LOGGER.warning(f"Conversion produced no steps for {game_file_path}")
        return None

    return GroundTruthTrajectory(
        task_id=task_id,
        task_type=task_type,
        pddl_params=pddl_params,
        steps=steps,
        raw_high_pddl=high_pddl,
    )


def get_ground_truth_step_count(
    game_file_path: str,
    split: str,
    alfworld_data: Optional[str] = None,
) -> Optional[int]:
    """Quick check for ground truth step count without full conversion.

    Args:
        game_file_path: Relative path like "look_at_obj_in_light-.../trial_..."
        split: Dataset split
        alfworld_data: Path to ALFWORLD_DATA

    Returns:
        Number of high-level steps or None if file not found
    """
    if alfworld_data is None:
        alfworld_data = os.environ.get(
            "ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld")
        )

    traj_data_path = (
        Path(alfworld_data) / "json_2.1.1" / split / game_file_path / "traj_data.json"
    )

    if not traj_data_path.exists():
        return None

    try:
        with open(traj_data_path, "r") as f:
            traj_data = json.load(f)

        high_pddl = traj_data.get("plan", {}).get("high_pddl", [])
        # Count non-NoOp actions
        return sum(
            1
            for action in high_pddl
            if action.get("discrete_action", {}).get("action", "")
            not in ("NoOp", "End", "")
        )
    except Exception:
        return None
