"""
Core VoyagerTrader implementation.

The main trading system that orchestrates the three key components:
- Automatic Curriculum
- Skill Library
- Iterative Prompting
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .curriculum import AutomaticCurriculum
from .prompting import IterativePrompting
from .skills import SkillLibrary


@dataclass
class TradingConfig:
    """Configuration for the VoyagerTrader system."""

    model_name: str = "gpt-4"
    max_iterations: int = 1000
    skill_library_path: str = "skills/"
    curriculum_temperature: float = 0.1
    enable_risk_management: bool = True
    max_position_size: float = 0.1  # 10% of portfolio per position


class VoyagerTrader:
    """
    Main VoyagerTrader system implementation.

    Combines automatic curriculum generation, skill library management,
    and iterative prompting to create an autonomous trading agent.
    """

    def __init__(self, config: Optional[TradingConfig] = None):
        """Initialize the VoyagerTrader system."""
        self.config = config or TradingConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize core components
        self.curriculum = AutomaticCurriculum(self.config)
        self.skill_library = SkillLibrary(self.config)
        self.prompting = IterativePrompting(self.config)

        # System state
        self.is_running = False
        self.current_task = None
        self.performance_metrics = {}

    def start(self) -> None:
        """Start the autonomous trading system."""
        self.logger.info("Starting VOYAGER-Trader system...")
        self.is_running = True

    def stop(self) -> None:
        """Stop the trading system."""
        self.logger.info("Stopping VOYAGER-Trader system...")
        self.is_running = False

    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "is_running": self.is_running,
            "current_task": self.current_task,
            "skills_learned": len(self.skill_library.skills),
            "performance": self.performance_metrics,
        }
