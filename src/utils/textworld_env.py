"""TextWorld environment wrapper for ALFWorld."""

import copy
import os
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from alfworld.agents.environment import get_environment
from textworld.envs.pddl import textgen as _tw_textgen

from .messaging import sanitize_action


def _patch_textworld_evalsymbol() -> None:
    """Ensure TextWorld EvalSymbol evaluates with the provided variables."""
    if getattr(_tw_textgen.EvalSymbol, "_agentify_patched", False):
        return

    _orig_derive = _tw_textgen.EvalSymbol.derive

    def _derive(self, context=None):
        context = context or self.context
        variables = context.get("variables", {})
        try:
            value = eval(self.expression, {}, variables)
        except Exception:
            return _orig_derive(self, context)
        return [_tw_textgen.TerminalSymbol(value)]

    _tw_textgen.EvalSymbol.derive = _derive
    _tw_textgen.EvalSymbol._agentify_patched = True


_patch_textworld_evalsymbol()


ASSETS_DIR = Path(__file__).resolve().parent.parent / "green_agent"
DEFAULT_TASK_LIST = ASSETS_DIR / "task_list.txt"


TEXTWORLD_ENV_CFG: Dict[str, Any] = {
    "dataset": {
        "data_path": "/home/jupyter/.cache/alfworld/json_2.1.1/train",
        "eval_id_data_path": "/home/jupyter/.cache/alfworld/json_2.1.1/valid_seen",
        "eval_ood_data_path": "/home/jupyter/.cache/alfworld/json_2.1.1/valid_unseen",
        "num_train_games": 0,
        "num_eval_games": 1,
    },
    "env": {
        "goal_desc_human_anns_prob": 0.0,
        "domain_randomization": False,
        "task_types": [1, 2, 3, 4, 5, 6],
        "expert_timeout_steps": 150,
        "expert_type": "planner",
    },
    "logic": {
        "domain": "/home/jupyter/.cache/alfworld/logic/alfred.pddl",
        "grammar": "/home/jupyter/.cache/alfworld/logic/alfred.twl2",
    },
    "general": {
        "training_method": "dqn",
    },
    "dagger": {
        "training": {
            "max_nb_steps_per_episode": 50,
        }
    },
    "rl": {
        "training": {
            "max_nb_steps_per_episode": 50,
        }
    },
}


class TextWorldEnvironmentError(Exception):
    """Raised when TextWorld environment operations fail."""


def load_task_from_list(
    task_index: int, task_list_path: Path = DEFAULT_TASK_LIST
) -> Dict[str, str]:
    """Load task metadata from task list file.

    Args:
        task_index: Index of task in list (0-based)
        task_list_path: Path to task list file

    Returns:
        Dictionary with task metadata (game_file_path, split, task_type, difficulty, description)
    """
    if not task_list_path.exists():
        raise TextWorldEnvironmentError(f"Task list not found at {task_list_path}")

    with task_list_path.open("r", encoding="utf-8") as handle:
        lines = [
            line.strip() for line in handle if line.strip() and not line.startswith("#")
        ]

    if not lines:
        raise TextWorldEnvironmentError(f"Task list {task_list_path} is empty")

    if task_index < 0 or task_index >= len(lines):
        raise TextWorldEnvironmentError(
            f"Task index {task_index} out of range (0-{len(lines) - 1})"
        )

    parts = lines[task_index].split("|")
    if len(parts) < 3:
        raise TextWorldEnvironmentError(
            f"Invalid task specification at index {task_index}: {lines[task_index]}"
        )

    return {
        "game_file_path": parts[0],
        "split": parts[1],
        "task_type": parts[2],
        "difficulty": parts[3] if len(parts) > 3 else "unknown",
        "description": parts[4] if len(parts) > 4 else "",
    }


@dataclass
class TaskConfig:
    """Configuration for a TextWorld evaluation task."""

    task_index: int = 0
    max_steps: int = 50
    task_list_path: Optional[str] = None
    episode_id: Optional[str] = None

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "TaskConfig":
        """Create TaskConfig from JSON payload."""
        return cls(
            task_index=int(payload.get("task_index", 0)),
            max_steps=int(payload.get("max_steps", 50)),
            task_list_path=payload.get("task_list_path"),
            episode_id=payload.get("episode_id"),
        )

    def resolve_task_list_path(self) -> Path:
        """Resolve task list path, using default if not specified."""
        if self.task_list_path:
            return Path(self.task_list_path)
        return DEFAULT_TASK_LIST


@dataclass
class StepResult:
    """Result of a single environment step."""

    observation: str
    reward: float
    done: bool
    cumulative_reward: float
    metadata: Dict[str, Any]


class TextWorldEnvironment:
    """Wrapper around ALFWorld environment for TextWorld evaluation."""

    def __init__(self, task_config: TaskConfig, *, use_expert_plan: bool = False):
        """Initialize environment with task configuration.

        Args:
            task_config: Configuration for the task
            use_expert_plan: Request expert plan from ALFWorld (for fast-mode runs)
        """
        self._task_config = task_config
        self._use_expert_plan = use_expert_plan
        self._env = None
        self._episode_id = (
            task_config.episode_id or f"ep_{int(time.time())}_{task_config.task_index}"
        )
        self._goal: str = ""
        self._current_observation: str = ""
        self._cumulative_reward = 0.0
        self._step_count = 0
        self._done = False
        self._info: Dict[str, Any] = {}
        self._start_time = time.time()
        self._task_entry: Optional[Dict[str, str]] = None
        self._expert_plan: list[str] = []

    @property
    def goal(self) -> str:
        """Get current goal description."""
        return self._goal

    @property
    def current_observation(self) -> str:
        """Get current observation."""
        return self._current_observation

    @property
    def episode_id(self) -> str:
        """Get episode ID."""
        return self._episode_id

    def setup(self) -> Dict[str, Any]:
        """Set up environment and load task.

        Returns:
            Dictionary with episode_id, task_entry, initial_observation, goal
        """
        def _as_str(val: Any) -> str:
            if isinstance(val, list):
                return " ".join(str(x) for x in val if x is not None).strip()
            return str(val).strip() if val is not None else ""

        def _is_generic_goal(text: str) -> bool:
            t = text.lower()
            # Heuristic: generic family names tend to be short and include 'task' or 'obj'
            if len(t) < 40 and (" task" in t or "task " in t):
                return True
            if " obj " in t or t.endswith(" obj") or "obj in light" in t:
                return True
            return False

        def _normalize_goal_text(info_map: Dict[str, Any], text: str) -> str:
            """Normalize goal to a specific, human-friendly imperative if possible."""
            goal = text.strip()
            low = goal.lower()

            # Remove trailing/leading 'task' wording
            goal = re.sub(r"\btask\b", "", goal, flags=re.IGNORECASE).strip(" .")

            # Replace generic 'obj/object' with a specific object if discoverable
            potential_keys = [
                "extra.goal_object",
                "extra.object",
                "extra.objects",
                "goal_object",
                "target_object",
                "object",
                "objects",
            ]

            # Helper to get nested keys like "extra.goal_object"
            def _get_nested(m: Dict[str, Any], dotted: str) -> Any:
                parts = dotted.split(".")
                cur: Any = m
                for p in parts:
                    if isinstance(cur, dict) and p in cur:
                        cur = cur[p]
                    else:
                        return None
                return cur

            obj = None
            for key in potential_keys:
                val = _get_nested(info_map, key)
                if isinstance(val, list) and val:
                    obj = str(val[0]).strip()
                elif isinstance(val, str) and val.strip():
                    obj = val.strip()
                if obj:
                    break

            if obj:
                # Common replacements
                goal = re.sub(r"\bobj(ect)?\b", obj, goal, flags=re.IGNORECASE)
                goal = goal.replace("Obj", obj).replace("OBJ", obj)

            # Prefer "under light" phrasing
            goal = re.sub(r"\bin light\b", "under light", goal, flags=re.IGNORECASE)
            goal = re.sub(r"\bwith\s+the\s+desklamp\b", "under light", goal, flags=re.IGNORECASE)

            # Normalize whitespace and capitalization minimally
            goal = re.sub(r"\s+", " ", goal).strip()
            return goal

        def _pick_best_goal(info_map: Dict[str, Any], task_entry_map: Dict[str, str]) -> str:
            # Candidate keys to try in order of typical specificity
            candidate_keys = [
                "extra.full_goal",
                "extra.human_goal",
                "extra.human_goals",
                "extra.goal",
                "goal",
                "extra.instruction",
                "instruction",
                "objective",
                "task_description",
                "extra.description",
                "description",
            ]

            # Flatten nested keys like "extra.goal"
            def _get_nested(m: Dict[str, Any], dotted: str) -> Any:
                parts = dotted.split(".")
                cur: Any = m
                for p in parts:
                    if isinstance(cur, dict) and p in cur:
                        cur = cur[p]
                    else:
                        return None
                return cur

            candidates: list[str] = []
            for key in candidate_keys:
                val = _get_nested(info_map, key)
                if val is None and key in ("description", "extra.description"):
                    val = task_entry_map.get("description")
                s = _as_str(val)
                if s:
                    candidates.append(s)

            # If nothing found, fallback to task entry description or a default
            if not candidates:
                fallback = task_entry_map.get("description") or "Complete the task"
                return _normalize_goal_text(info_map, fallback)

            # Prefer non-generic, then choose longest
            non_generic = [c for c in candidates if not _is_generic_goal(c)]
            if non_generic:
                best = max(non_generic, key=len)
                return _normalize_goal_text(info_map, best)
            best = max(candidates, key=len)
            return _normalize_goal_text(info_map, best)

        task_entry = load_task_from_list(
            self._task_config.task_index, self._task_config.resolve_task_list_path()
        )
        self._task_entry = task_entry

        alfworld_data = os.environ.get("ALFWORLD_DATA") or os.path.expanduser(
            "~/.cache/alfworld"
        )
        game_dir = (
            Path(alfworld_data)
            / "json_2.1.1"
            / task_entry["split"]
            / task_entry["game_file_path"]
        )
        game_file_path = game_dir / "game.tw-pddl"
        if not game_file_path.exists():
            raise TextWorldEnvironmentError(
                f"Game file not found: {game_file_path}. "
                "Ensure ALFWorld assets are available."
            )

        config = copy.deepcopy(TEXTWORLD_ENV_CFG)
        if self._use_expert_plan:
            # Request expert planner commands for fast mode.
            config["general"]["training_method"] = "dagger"
            config["dagger"]["training"]["max_nb_steps_per_episode"] = self._task_config.max_steps
            config["env"]["expert_type"] = "planner"

        config["dataset"] = {
            "data_path": str(game_dir),
            "eval_id_data_path": str(game_dir),
            "eval_ood_data_path": str(game_dir),
            "num_train_games": -1,
            "num_eval_games": -1,
        }

        env_cls = get_environment("AlfredTWEnv")
        self._env = env_cls(config=config).init_env(batch_size=1)

        observations, info = self._env.reset()
        observation = observations[0]

        # Extract goal with heuristics favoring specific over family name
        goal_value = _pick_best_goal(info, task_entry)

        # Initialize state
        self._goal = goal_value
        self._current_observation = observation
        self._cumulative_reward = 0.0
        self._step_count = 0
        self._done = False
        self._info = {
            key: value[0] if isinstance(value, list) else value
            for key, value in info.items()
        }
        # Capture expert plan if available (fast-mode).
        self._update_expert_plan(info)
        self._start_time = time.time()

        return {
            "episode_id": self._episode_id,
            "task_entry": task_entry,
            "initial_observation": observation,
            "goal": self._goal,
        }

    def step(self, raw_action: str) -> StepResult:
        """Execute an action in the environment.

        Args:
            raw_action: Action string from agent

        Returns:
            StepResult with observation, reward, done flag, etc.
        """
        if self._env is None:
            raise TextWorldEnvironmentError(
                "Environment not initialized; call setup() first"
            )

        action = sanitize_action(raw_action)

        observations, rewards, done_flags, info = self._env.step([action])
        observation = observations[0]
        reward = float(rewards[0])
        done = bool(done_flags[0])

        # Unbatch info
        info_unbatched: Dict[str, Any] = {}
        for key, value in info.items():
            if isinstance(value, list) and value:
                info_unbatched[key] = value[0]
            else:
                info_unbatched[key] = value

        # Update state
        self._current_observation = observation
        self._cumulative_reward += reward
        self._step_count += 1
        self._done = done
        self._info = info_unbatched
        self._update_expert_plan(info_unbatched)

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            cumulative_reward=self._cumulative_reward,
            metadata=info_unbatched,
        )

    def metrics(self) -> Dict[str, Any]:
        """Get evaluation metrics.

        Returns:
            Dictionary with success, step_count, cumulative_reward, etc.
        """
        return {
            "episode_id": self._episode_id,
            "step_count": self._step_count,
            "cumulative_reward": self._cumulative_reward,
            "success": bool(self._info.get("won", False)),
        }

    def walkthrough(self) -> list[str]:
        """Return expert plan if available, else empty list."""
        return list(self._expert_plan) if self._expert_plan else []

    def reset(self) -> None:
        """Reset and clean up environment."""
        if self._env is not None:
            self._env.close()
        self._env = None

    def _update_expert_plan(self, info: Optional[Dict[str, Any]]) -> None:
        """Normalize and store expert plan if present in info."""
        if not self._use_expert_plan:
            return
        plan = info.get("extra.expert_plan") if isinstance(info, dict) else None
        self._expert_plan = []
        if isinstance(plan, list):
            for item in plan:
                if isinstance(item, list):
                    self._expert_plan.extend([str(x) for x in item if x is not None])
                elif item is not None:
                    self._expert_plan.append(str(item))
