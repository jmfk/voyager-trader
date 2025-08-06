"""
Skill Library implementation.

Stores and manages learned trading skills, similar to VOYAGER's
approach to skill acquisition and reuse.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class TradingSkill:
    """Represents a learned trading skill."""

    name: str
    description: str
    code: str
    performance_metrics: Dict[str, float]
    usage_count: int = 0
    success_rate: float = 0.0
    prerequisites: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


class SkillLibrary:
    """
    Library for storing and retrieving learned trading skills.

    Manages skill acquisition, composition, and reuse patterns
    similar to VOYAGER's skill library approach.
    """

    def __init__(self, config):
        """Initialize the skill library."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.skills: Dict[str, TradingSkill] = {}
        self.skill_dependencies: Dict[str, List[str]] = {}

        # Load existing skills if library path exists
        self.library_path = Path(config.skill_library_path)
        self.library_path.mkdir(exist_ok=True)
        self._load_existing_skills()

    def add_skill(self, skill: TradingSkill) -> None:
        """Add a new skill to the library."""
        self.skills[skill.name] = skill
        self.logger.info(f"Added skill: {skill.name}")
        self._save_skill(skill)

    def get_skill(self, name: str) -> Optional[TradingSkill]:
        """Retrieve a skill by name."""
        return self.skills.get(name)

    def search_skills(
        self, tags: List[str] = None, min_success_rate: float = 0.0
    ) -> List[TradingSkill]:
        """Search for skills matching criteria."""
        results = []

        for skill in self.skills.values():
            # Check success rate
            if skill.success_rate < min_success_rate:
                continue

            # Check tags if specified
            if tags and not any(tag in skill.tags for tag in tags):
                continue

            results.append(skill)

        # Sort by success rate and usage count
        results.sort(key=lambda s: (s.success_rate, s.usage_count), reverse=True)
        return results

    def compose_skills(self, skill_names: List[str]) -> Optional[str]:
        """Compose multiple skills into a new strategy."""
        skills = [self.get_skill(name) for name in skill_names]

        if not all(skills):
            self.logger.error("One or more skills not found for composition")
            return None

        # Simple composition - concatenate skill code
        composed_code = "\n\n".join(skill.code for skill in skills)
        return composed_code

    def update_skill_performance(
        self, name: str, performance: Dict[str, float]
    ) -> None:
        """Update performance metrics for a skill."""
        if name in self.skills:
            skill = self.skills[name]
            skill.performance_metrics.update(performance)
            skill.usage_count += 1

            # Update success rate based on performance
            if "success" in performance:
                old_rate = skill.success_rate
                skill.success_rate = (
                    old_rate * (skill.usage_count - 1) + performance["success"]
                ) / skill.usage_count

            self._save_skill(skill)

    def _load_existing_skills(self) -> None:
        """Load skills from the library directory."""
        for skill_file in self.library_path.glob("*.json"):
            try:
                with open(skill_file, "r") as f:
                    data = json.load(f)
                    skill = TradingSkill(**data)
                    self.skills[skill.name] = skill
            except (json.JSONDecodeError, KeyError, TypeError, OSError) as e:
                self.logger.error(f"Failed to load skill from {skill_file}: {e}")

    def _save_skill(self, skill: TradingSkill) -> None:
        """Save a skill to the library directory."""
        try:
            skill_file = self.library_path / f"{skill.name}.json"
            with open(skill_file, "w") as f:
                # Convert skill to dict for JSON serialization
                skill_dict = {
                    "name": skill.name,
                    "description": skill.description,
                    "code": skill.code,
                    "performance_metrics": skill.performance_metrics,
                    "usage_count": skill.usage_count,
                    "success_rate": skill.success_rate,
                    "prerequisites": skill.prerequisites,
                    "tags": skill.tags,
                }
                json.dump(skill_dict, f, indent=2)
        except (OSError, json.JSONEncodeError, TypeError) as e:
            self.logger.error(f"Failed to save skill {skill.name}: {e}")
