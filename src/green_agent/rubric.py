"""Evaluation rubric dataclasses and loading logic."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import tomllib


@dataclass
class CriterionConfig:
    """Configuration for a single evaluation criterion."""

    name: str
    weight: float
    description: str
    scoring_guidelines: Dict[str, str]
    examples: Dict[str, str]
    prompt_template: str

    @classmethod
    def from_dict(cls, key: str, data: Dict[str, Any]) -> CriterionConfig:
        """Create CriterionConfig from dictionary."""
        return cls(
            name=data.get("name", key),
            weight=float(data.get("weight", 0.0)),
            description=data.get("description", ""),
            scoring_guidelines=data.get("scoring_guidelines", {}),
            examples=data.get("examples", {}),
            prompt_template=data.get("prompt_template", ""),
        )

    def validate(self) -> List[str]:
        """Validate criterion configuration and return list of errors."""
        errors = []
        if not self.name:
            errors.append(f"Criterion '{self.name}': name is required")
        if not 0.0 <= self.weight <= 1.0:
            errors.append(
                f"Criterion '{self.name}': weight must be between 0.0 and 1.0, got {self.weight}"
            )
        if not self.description:
            errors.append(f"Criterion '{self.name}': description is required")
        if not self.prompt_template:
            errors.append(f"Criterion '{self.name}': prompt_template is required")
        return errors


@dataclass
class ScoringScale:
    """Scoring scale configuration."""

    min: float = 1.0
    max: float = 10.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ScoringScale:
        """Create ScoringScale from dictionary."""
        return cls(
            min=float(data.get("min", 1.0)),
            max=float(data.get("max", 10.0)),
        )

    def validate(self) -> List[str]:
        """Validate scoring scale and return list of errors."""
        errors = []
        if self.min >= self.max:
            errors.append(
                f"Scoring scale: min ({self.min}) must be less than max ({self.max})"
            )
        if self.min < 0:
            errors.append(f"Scoring scale: min ({self.min}) must be non-negative")
        return errors


@dataclass
class ModelSettings:
    """LLM model settings for evaluation."""

    model: str
    temperature: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelSettings:
        """Create ModelSettings from dictionary."""
        return cls(
            model=str(data.get("model", "openai/gpt-4o")),
            temperature=float(data.get("temperature", 0.3)),
        )


@dataclass
class EvaluationRubric:
    """Evaluation rubric with criteria, weights, and configuration."""

    criteria: Dict[str, CriterionConfig] = field(default_factory=dict)
    scoring_scale: ScoringScale = field(default_factory=lambda: ScoringScale())
    qualitative_assessments: Dict[str, Any] = field(default_factory=dict)
    reasoning_trace_analysis: Dict[str, Any] = field(default_factory=dict)
    model_settings: Dict[str, ModelSettings] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvaluationRubric:
        """Create EvaluationRubric from dictionary."""
        # Load criteria
        criteria = {}
        criteria_data = data.get("criteria", {})
        for key, criterion_data in criteria_data.items():
            criteria[key] = CriterionConfig.from_dict(key, criterion_data)

        # Load scoring scale
        scoring_scale = ScoringScale.from_dict(data.get("scoring_scale", {}))

        # Load qualitative assessments config
        qualitative_assessments = data.get("qualitative_assessments", {})
        reasoning_trace_analysis = data.get("reasoning_trace_analysis", {})

        # Load model settings
        model_settings = {}
        model_settings_data = data.get("model_settings", {})
        for key, settings_data in model_settings_data.items():
            model_settings[key] = ModelSettings.from_dict(settings_data)

        return cls(
            criteria=criteria,
            scoring_scale=scoring_scale,
            qualitative_assessments=qualitative_assessments,
            reasoning_trace_analysis=reasoning_trace_analysis,
            model_settings=model_settings,
        )

    def validate(self) -> List[str]:
        """Validate rubric configuration and return list of errors."""
        errors = []

        # Validate scoring scale
        errors.extend(self.scoring_scale.validate())

        # Validate criteria
        if not self.criteria:
            errors.append("No criteria defined in rubric")
        else:
            for criterion in self.criteria.values():
                errors.extend(criterion.validate())

        # Validate weights sum to ~1.0
        total_weight = sum(c.weight for c in self.criteria.values())
        if abs(total_weight - 1.0) > 0.01:
            errors.append(
                f"Criterion weights sum to {total_weight:.3f}, should be approximately 1.0"
            )

        return errors

    def get_criterion_names(self) -> List[str]:
        """Get list of criterion names."""
        return list(self.criteria.keys())

    def get_total_weight(self) -> float:
        """Get sum of all criterion weights."""
        return sum(c.weight for c in self.criteria.values())


def load_rubric_from_json(file_path: Path) -> EvaluationRubric:
    """Load evaluation rubric from JSON file."""
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return EvaluationRubric.from_dict(data)


def load_rubric_from_toml(file_path: Path) -> EvaluationRubric:
    """Load evaluation rubric from TOML file."""
    with file_path.open("rb") as f:
        data = tomllib.load(f)
    return EvaluationRubric.from_dict(data)


def load_rubric(file_path: Optional[Path] = None) -> EvaluationRubric:
    """Load evaluation rubric from file.

    Looks for rubric in this order:
    1. file_path if provided
    2. TEXTWORLD_EVALUATION_RUBRIC_PATH environment variable
    3. Default location: src/green_agent/evaluation_rubric.json

    Args:
        file_path: Optional explicit path to rubric file

    Returns:
        EvaluationRubric instance

    Raises:
        FileNotFoundError: If rubric file cannot be found
        ValueError: If rubric file is invalid
    """
    # Determine which file to load
    if file_path is not None:
        rubric_path = Path(file_path)
    else:
        env_path = os.environ.get("TEXTWORLD_EVALUATION_RUBRIC_PATH")
        if env_path:
            rubric_path = Path(env_path)
        else:
            # Default location
            rubric_path = Path(__file__).parent / "evaluation_rubric.json"

    if not rubric_path.exists():
        raise FileNotFoundError(f"Rubric file not found: {rubric_path}")

    # Load based on file extension
    if rubric_path.suffix == ".json":
        rubric = load_rubric_from_json(rubric_path)
    elif rubric_path.suffix == ".toml":
        rubric = load_rubric_from_toml(rubric_path)
    else:
        # Try JSON first, then TOML
        try:
            rubric = load_rubric_from_json(rubric_path)
        except json.JSONDecodeError:
            rubric = load_rubric_from_toml(rubric_path)

    # Validate rubric
    errors = rubric.validate()
    if errors:
        error_msg = "Rubric validation failed:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise ValueError(error_msg)

    return rubric


__all__ = [
    "CriterionConfig",
    "ScoringScale",
    "ModelSettings",
    "EvaluationRubric",
    "load_rubric",
    "load_rubric_from_json",
    "load_rubric_from_toml",
]
