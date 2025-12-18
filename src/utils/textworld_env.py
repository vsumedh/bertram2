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
from .ground_truth_baseline import (
    GroundTruthTrajectory,
    load_ground_truth_trajectory,
)


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


def _build_default_config() -> Dict[str, Any]:
    """Build config with paths from environment.

    Uses ALFWORLD_DATA environment variable if set, otherwise defaults to ~/.cache/alfworld.
    """
    alfworld_data = os.environ.get(
        "ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld")
    )

    return {
        "dataset": {
            "data_path": f"{alfworld_data}/json_2.1.1/train",
            "eval_id_data_path": f"{alfworld_data}/json_2.1.1/valid_seen",
            "eval_ood_data_path": f"{alfworld_data}/json_2.1.1/valid_unseen",
            "num_train_games": 0,
            "num_eval_games": 1,
        },
        "env": {
            "goal_desc_human_anns_prob": 0.0,
            "domain_randomization": False,
            "task_types": [1, 2, 3, 4, 5, 6],
            "expert_timeout_steps": 150,
            # Use handcoded expert (domain-aware heuristics) instead of PDDL planner.
            # The handcoded expert is ALFWorld's recommended default and provides
            # a meaningful baseline representing competent human-level performance.
            "expert_type": "handcoded",
        },
        "logic": {
            "domain": f"{alfworld_data}/logic/alfred.pddl",
            "grammar": f"{alfworld_data}/logic/alfred.twl2",
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


TEXTWORLD_ENV_CFG = _build_default_config()


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
    admissible_commands: list[str] = None  # Valid actions for current state


@dataclass
class ExpertTrajectoryStep:
    """A single step from the expert trajectory."""

    step_num: int
    action: str
    observation: str
    reward: float
    done: bool


@dataclass
class ExpertTrajectory:
    """Complete expert trajectory for an episode."""

    episode_id: str
    goal: str
    initial_observation: str
    steps: list[ExpertTrajectoryStep]
    success: bool
    total_reward: float

    @property
    def actions(self) -> list[str]:
        """Get list of expert actions."""
        return [s.action for s in self.steps]

    @property
    def observations(self) -> list[str]:
        """Get list of observations (including initial)."""
        return [self.initial_observation] + [s.observation for s in self.steps]

    @property
    def step_count(self) -> int:
        """Get total number of steps."""
        return len(self.steps)


class TextWorldEnvironment:
    """Wrapper around ALFWorld environment for TextWorld evaluation."""

    def __init__(self, task_config: TaskConfig, *, use_expert_plan: bool = False):
        """Initialize environment with task configuration.

        Args:
            task_config: Configuration for the task
            use_expert_plan: Request expert plan from ALFWorld (for evaluation metrics)
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
        self._admissible_commands: list[str] = []

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

    @property
    def admissible_commands(self) -> list[str]:
        """Get admissible commands for current state.

        Returns list of valid actions the agent can take in the current state.
        This is provided directly by the ALFWorld simulator.
        """
        return self._admissible_commands

    def setup(self) -> Dict[str, Any]:
        """Set up environment and load task.

        Returns:
            Dictionary with episode_id, task_entry, initial_observation, goal
        """
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
            # Request handcoded expert for generating evaluation baselines.
            # The handcoded expert uses domain-aware heuristics (knows where objects
            # are typically located) and represents competent human-level performance.
            config["general"]["training_method"] = "dagger"
            config["dagger"]["training"]["max_nb_steps_per_episode"] = (
                self._task_config.max_steps
            )
            config["env"]["expert_type"] = "handcoded"

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

        # Extract goal from observation: ALFWorld embeds "Your task is to: <task>"
        match = re.search(
            r"Your task is to:\s*(.+?)(?:\n|$)", observation, re.IGNORECASE
        )
        goal_value = match.group(1).strip() if match else ""

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
        # Extract admissible commands from ALFWorld simulator
        raw_admissible = info.get("admissible_commands", [])
        if raw_admissible and isinstance(raw_admissible[0], list):
            # Unbatch: take first element if batched
            self._admissible_commands = list(raw_admissible[0])
        elif raw_admissible:
            self._admissible_commands = list(raw_admissible)
        else:
            self._admissible_commands = []
        # Capture expert plan if available (for evaluation metrics).
        self._update_expert_plan(info)
        self._start_time = time.time()

        return {
            "episode_id": self._episode_id,
            "task_entry": task_entry,
            "initial_observation": observation,
            "goal": self._goal,
            "admissible_commands": self._admissible_commands,
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

        # Extract admissible commands for current state
        raw_admissible = info.get("admissible_commands", [])
        if raw_admissible and isinstance(raw_admissible[0], list):
            self._admissible_commands = list(raw_admissible[0])
        elif raw_admissible:
            self._admissible_commands = list(raw_admissible)
        else:
            self._admissible_commands = []

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
            admissible_commands=self._admissible_commands,
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

    def get_ground_truth_trajectory(self) -> GroundTruthTrajectory:
        """Get the optimal ground truth trajectory from traj_data.json.

        This loads the pre-computed optimal trajectory from the ALFRED dataset,
        which represents actual human demonstrations. This is the ideal baseline
        for evaluation.

        Must be called after setup() to have access to task_entry and observation.

        Returns:
            GroundTruthTrajectory with optimal TextWorld commands

        Raises:
            TextWorldEnvironmentError: If setup not called or trajectory unavailable
        """
        if self._task_entry is None:
            raise TextWorldEnvironmentError(
                "Environment not initialized; call setup() first"
            )

        game_file_path = self._task_entry["game_file_path"]
        split = self._task_entry["split"]

        trajectory = load_ground_truth_trajectory(
            game_file_path=game_file_path,
            split=split,
            initial_observation=self._current_observation,
        )

        if trajectory is None:
            raise TextWorldEnvironmentError(
                f"Ground truth trajectory not found for {game_file_path}"
            )

        return trajectory

    def run_expert(self, max_steps: int = 100) -> ExpertTrajectory:
        """Execute expert policy to completion and return full trajectory.

        Must be called immediately after setup() before any agent steps.
        The environment will be at the terminal state after this call.

        Args:
            max_steps: Maximum steps to run expert (safety limit)

        Returns:
            ExpertTrajectory with complete action/observation sequence

        Raises:
            TextWorldEnvironmentError: If environment not set up or expert unavailable
        """
        if self._env is None:
            raise TextWorldEnvironmentError(
                "Environment not initialized; call setup() first"
            )
        if not self._use_expert_plan:
            raise TextWorldEnvironmentError(
                "Expert plan not enabled; set use_expert_plan=True in constructor"
            )

        steps: list[ExpertTrajectoryStep] = []
        initial_obs = self._current_observation
        step_num = 0

        while not self._done and step_num < max_steps:
            # Get next expert action
            expert_plan = self.walkthrough()
            if not expert_plan:
                # No more expert actions but not done - unusual state
                break

            expert_action = expert_plan[0]

            # Execute expert action
            result = self.step(expert_action)
            step_num += 1

            steps.append(
                ExpertTrajectoryStep(
                    step_num=step_num,
                    action=expert_action,
                    observation=result.observation,
                    reward=result.reward,
                    done=result.done,
                )
            )

            if result.done:
                break

        return ExpertTrajectory(
            episode_id=self._episode_id,
            goal=self._goal,
            initial_observation=initial_obs,
            steps=steps,
            success=bool(self._info.get("won", False)),
            total_reward=self._cumulative_reward,
        )

    def reset(self) -> None:
        """Reset and clean up environment."""
        if self._env is not None:
            self._env.close()
        self._env = None

    def reset_to_initial_state(self) -> Dict[str, Any]:
        """Reset environment to initial state of current episode.

        This allows re-running the same episode after running the expert.
        Must have previously called setup().

        Returns:
            Same format as setup() - episode_id, task_entry, initial_observation, goal
        """
        if self._env is None:
            raise TextWorldEnvironmentError(
                "Environment not initialized; call setup() first"
            )

        # Reset the gym environment
        observations, info = self._env.reset()
        observation = observations[0]

        # Re-initialize state
        self._current_observation = observation
        self._cumulative_reward = 0.0
        self._step_count = 0
        self._done = False
        self._info = {
            key: value[0] if isinstance(value, list) else value
            for key, value in info.items()
        }

        # Re-extract admissible commands
        raw_admissible = info.get("admissible_commands", [])
        if raw_admissible and isinstance(raw_admissible[0], list):
            self._admissible_commands = list(raw_admissible[0])
        elif raw_admissible:
            self._admissible_commands = list(raw_admissible)
        else:
            self._admissible_commands = []

        self._update_expert_plan(info)
        self._start_time = time.time()

        return {
            "episode_id": self._episode_id,
            "task_entry": self._task_entry,
            "initial_observation": observation,
            "goal": self._goal,
            "admissible_commands": self._admissible_commands,
        }

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
