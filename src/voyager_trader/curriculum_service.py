"""
Curriculum Service with persistence and resumption functionality.

This module provides the main CurriculumService that orchestrates all curriculum
components and handles state persistence and recovery.
"""

import json
import logging
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from .curriculum import AutomaticCurriculumService
from .curriculum_components import (
    AdaptiveLogicEngine,
    BasicCurriculumGenerator,
    MarketContextAnalyzer,
    PerformanceProgressTracker,
    StandardDifficultyAssessor,
    sanitize_file_path,
)
from .models.system import Agent, Curriculum, Environment, Task


class CurriculumPersistenceService:
    """
    Service for persisting and recovering curriculum state.

    Handles saving curriculum state, task history, and agent progress
    to enable system restart and recovery.
    """

    def __init__(
        self, storage_path: str = "curriculum_data", max_history_entries: int = 1000
    ):
        """Initialize with storage configuration."""
        # Sanitize storage path for security
        sanitized_path = sanitize_file_path(storage_path)
        self.storage_path = sanitized_path
        self.storage_path.mkdir(exist_ok=True)
        self.max_history_entries = max_history_entries
        self.logger = logging.getLogger(__name__)

    def save_curriculum_state(
        self, agent: Agent, curriculum: Curriculum, active_tasks: List[Task] = None
    ) -> None:
        """Save complete curriculum state for an agent."""
        self.logger.info(f"Saving curriculum state for agent {agent.name}")

        state = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self._serialize_agent(agent),
            "curriculum": self._serialize_curriculum(curriculum),
            "active_tasks": [
                self._serialize_task(task) for task in (active_tasks or [])
            ],
            "version": "1.0",
        }

        # Save to agent-specific file
        state_file = self.storage_path / f"{agent.id}_curriculum_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

        self.logger.info(f"Saved curriculum state to {state_file}")

    def load_curriculum_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Load curriculum state for an agent."""
        state_file = self.storage_path / f"{agent_id}_curriculum_state.json"

        if not state_file.exists():
            self.logger.info(f"No saved state found for agent {agent_id}")
            return None

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            self.logger.info(f"Loaded curriculum state for agent {agent_id}")
            return state

        except (OSError, json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Failed to load curriculum state: {e}")
            return None

    def save_task_completion(
        self, agent_id: str, task: Task, success: bool, metrics: Dict[str, Any]
    ) -> None:
        """Save task completion record."""
        completion_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": agent_id,
            "task": self._serialize_task(task),
            "success": success,
            "metrics": metrics,
        }

        # Append to task history file
        history_file = self.storage_path / f"{agent_id}_task_history.jsonl"
        with open(history_file, "a") as f:
            f.write(json.dumps(completion_record) + "\n")

        self.logger.info(f"Saved task completion for {task.title}")

    def load_task_history(self, agent_id: str) -> List[Dict[str, Any]]:
        """Load task completion history for an agent."""
        history_file = self.storage_path / f"{agent_id}_task_history.jsonl"

        if not history_file.exists():
            return []

        history = []
        try:
            with open(history_file, "r") as f:
                for line in f:
                    if line.strip():
                        history.append(json.loads(line))

            self.logger.info(f"Loaded {len(history)} task completion records")
            return history

        except (OSError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to load task history: {e}")
            return []

    def clear_agent_data(self, agent_id: str) -> None:
        """Clear all persisted data for an agent."""
        files_to_remove = [
            self.storage_path / f"{agent_id}_curriculum_state.json",
            self.storage_path / f"{agent_id}_task_history.jsonl",
        ]

        for file_path in files_to_remove:
            if file_path.exists():
                file_path.unlink()
                self.logger.info(f"Removed {file_path}")

    def _serialize_agent(self, agent: Agent) -> Dict[str, Any]:
        """Serialize agent for persistence."""
        return {
            "id": agent.id,
            "name": agent.name,
            "version": agent.version,
            "state": agent.state.value,
            "learned_skills": agent.learned_skills,
            "active_tasks": agent.active_tasks,
            "completed_tasks": agent.completed_tasks,
            "performance_metrics": {
                k: float(v) for k, v in agent.performance_metrics.items()
            },
            "experience_count": agent.experience_count,
            "knowledge_items": agent.knowledge_items,
            "total_trades": agent.total_trades,
            "successful_trades": agent.successful_trades,
            "total_pnl": {
                "amount": float(agent.total_pnl.amount),
                "currency": str(agent.total_pnl.currency),
            },
        }

    def _serialize_curriculum(self, curriculum: Curriculum) -> Dict[str, Any]:
        """Serialize curriculum for persistence."""
        return {
            "id": curriculum.id,
            "name": curriculum.name,
            "agent_id": curriculum.agent_id,
            "strategy": curriculum.strategy.value,
            "current_difficulty": curriculum.current_difficulty.value,
            "target_skills": curriculum.target_skills,
            "completed_skills": curriculum.completed_skills,
            "active_tasks": curriculum.active_tasks,
            "task_queue": curriculum.task_queue,
            "completed_tasks": curriculum.completed_tasks,
            "failed_tasks": curriculum.failed_tasks,
            "progress_metrics": {
                k: float(v) for k, v in curriculum.progress_metrics.items()
            },
            "generation_parameters": curriculum.generation_parameters,
        }

    def _serialize_task(self, task: Task) -> Dict[str, Any]:
        """Serialize task for persistence."""
        return {
            "id": task.id,
            "title": task.title,
            "description": task.description,
            "task_type": task.task_type.value,
            "priority": task.priority.value,
            "difficulty": task.difficulty.value,
            "status": task.status.value,
            "objectives": task.objectives,
            "success_criteria": task.success_criteria,
            "required_skills": task.required_skills,
            "parameters": task.parameters,
            "progress_percentage": float(task.progress_percentage),
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": (
                task.completed_at.isoformat() if task.completed_at else None
            ),
        }

    def cleanup_old_files(self) -> None:
        """Clean up old files and implement file rotation policies."""
        self.logger.info("Starting file cleanup and rotation")

        try:
            # Rotate task history files if they exceed max entries
            for history_file in self.storage_path.glob("*_task_history.jsonl"):
                self._rotate_history_file(history_file)

            # Clean up temporary files older than 7 days
            import time

            cutoff_time = time.time() - (7 * 24 * 60 * 60)  # 7 days ago

            for temp_file in self.storage_path.glob("*.tmp"):
                if temp_file.stat().st_mtime < cutoff_time:
                    temp_file.unlink()
                    self.logger.info(f"Removed old temp file: {temp_file}")

            # Clean up old backup files (keep only latest 5)
            for agent_dir in self.storage_path.glob("*_curriculum_state.json.bak.*"):
                backup_files = sorted(
                    self.storage_path.glob(f"{agent_dir.stem}.bak.*"),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True,
                )
                for old_backup in backup_files[5:]:  # Keep only 5 most recent
                    old_backup.unlink()
                    self.logger.info(f"Removed old backup: {old_backup}")

        except (OSError, IOError) as e:
            self.logger.error(f"Error during file cleanup: {e}")

    def _rotate_history_file(self, history_file: Path) -> None:
        """Rotate history file if it exceeds maximum entries."""
        try:
            # Count lines in file
            with open(history_file, "r") as f:
                line_count = sum(1 for _ in f)

            if line_count > self.max_history_entries:
                self.logger.info(
                    f"Rotating history file {history_file} ({line_count} entries)"
                )

                # Read last max_history_entries lines
                with open(history_file, "r") as f:
                    lines = f.readlines()

                # Create backup
                import time as time_module

                backup_file = history_file.with_suffix(
                    f".bak.{int(time_module.time())}"
                )
                history_file.rename(backup_file)

                # Write truncated history
                keep_lines = lines[-self.max_history_entries :]
                with open(history_file, "w") as f:
                    f.writelines(keep_lines)

                self.logger.info(
                    f"Rotated {history_file}, kept {len(keep_lines)} entries"
                )

        except (OSError, IOError) as e:
            self.logger.error(f"Error rotating history file {history_file}: {e}")

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about storage usage."""
        stats = {
            "storage_path": str(self.storage_path),
            "total_files": 0,
            "total_size_bytes": 0,
            "file_types": {},
            "oldest_file": None,
            "newest_file": None,
        }

        try:
            oldest_time = float("inf")
            newest_time = 0

            for file_path in self.storage_path.rglob("*"):
                if file_path.is_file():
                    stats["total_files"] += 1
                    file_size = file_path.stat().st_size
                    stats["total_size_bytes"] += file_size

                    # Track file types
                    suffix = file_path.suffix or "no_extension"
                    stats["file_types"][suffix] = stats["file_types"].get(suffix, 0) + 1

                    # Track oldest/newest
                    mtime = file_path.stat().st_mtime
                    if mtime < oldest_time:
                        oldest_time = mtime
                        stats["oldest_file"] = str(file_path)
                    if mtime > newest_time:
                        newest_time = mtime
                        stats["newest_file"] = str(file_path)

            # Convert to human readable
            stats["total_size_mb"] = round(stats["total_size_bytes"] / (1024 * 1024), 2)

        except (OSError, IOError) as e:
            self.logger.error(f"Error collecting storage stats: {e}")

        return stats


class ResumableCurriculumService(AutomaticCurriculumService):
    """
    Enhanced curriculum service with persistence and resumption capabilities.

    Extends the base AutomaticCurriculumService to add state persistence,
    recovery from interruptions, and curriculum resumption.
    """

    def __init__(self, config: Dict[str, Any], storage_path: str = "curriculum_data"):
        """Initialize with persistence capability."""
        # Initialize concrete implementations
        generator = BasicCurriculumGenerator(config)
        difficulty_assessor = StandardDifficultyAssessor(config)
        progress_tracker = PerformanceProgressTracker(config)
        adaptive_engine = AdaptiveLogicEngine(config)
        context_analyzer = MarketContextAnalyzer(config)

        # Initialize parent
        super().__init__(
            generator=generator,
            difficulty_assessor=difficulty_assessor,
            progress_tracker=progress_tracker,
            adaptive_engine=adaptive_engine,
            context_analyzer=context_analyzer,
            config=config,
        )

        # Add persistence
        self.persistence = CurriculumPersistenceService(storage_path)
        self.auto_save = config.get("auto_save", True)
        self.save_interval_tasks = config.get("save_interval_tasks", 1)
        self.completed_tasks_count = 0

    def initialize_or_resume_curriculum(
        self, agent: Agent, environment: Environment
    ) -> tuple[Agent, Curriculum, List[Task]]:
        """
        Initialize new curriculum or resume from saved state.

        Returns:
            Tuple of (agent, curriculum, active_tasks)
        """
        self.logger.info(f"Initializing curriculum for agent {agent.name}")

        # Try to load saved state
        saved_state = self.persistence.load_curriculum_state(agent.id)

        if saved_state:
            self.logger.info("Resuming from saved curriculum state")
            return self._resume_from_state(saved_state, agent)
        else:
            self.logger.info("Initializing new curriculum")
            return self._initialize_new_curriculum(agent, environment)

    def generate_next_task_with_persistence(
        self, agent: Agent, curriculum: Curriculum, environment: Environment
    ) -> Optional[Task]:
        """Generate next task and optionally save state."""
        task = self.generate_next_task(agent, curriculum, environment)

        if task and self.auto_save:
            self.persistence.save_curriculum_state(agent, curriculum, [task])

        return task

    def complete_task_with_persistence(
        self,
        task: Task,
        agent: Agent,
        curriculum: Curriculum,
        success: bool,
        metrics: Dict[str, Any],
    ) -> Curriculum:
        """Complete task and save state."""
        # Complete the task
        updated_curriculum = self.complete_task(
            task, agent, curriculum, success, metrics
        )

        # Save completion record
        self.persistence.save_task_completion(agent.id, task, success, metrics)

        # Auto-save state if configured
        self.completed_tasks_count += 1
        if (
            self.auto_save
            and self.completed_tasks_count % self.save_interval_tasks == 0
        ):
            self.persistence.save_curriculum_state(agent, updated_curriculum)

        return updated_curriculum

    def save_current_state(
        self, agent: Agent, curriculum: Curriculum, active_tasks: List[Task] = None
    ) -> None:
        """Manually save current curriculum state."""
        self.persistence.save_curriculum_state(agent, curriculum, active_tasks or [])

    def get_curriculum_history(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive curriculum history for an agent."""
        task_history = self.persistence.load_task_history(agent_id)

        # Analyze history
        total_tasks = len(task_history)
        successful_tasks = sum(1 for record in task_history if record["success"])
        success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0

        # Group by difficulty level
        difficulty_stats = {}
        for record in task_history:
            difficulty = record["task"]["difficulty"]
            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = {"total": 0, "successful": 0}
            difficulty_stats[difficulty]["total"] += 1
            if record["success"]:
                difficulty_stats[difficulty]["successful"] += 1

        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": success_rate,
            "difficulty_stats": difficulty_stats,
            "recent_tasks": task_history[-10:] if task_history else [],
            "task_history": task_history,
        }

    def cleanup_agent_data(self, agent_id: str) -> None:
        """Clean up all persisted data for an agent."""
        self.persistence.clear_agent_data(agent_id)
        self.logger.info(f"Cleaned up all data for agent {agent_id}")

    def _resume_from_state(
        self, saved_state: Dict[str, Any], current_agent: Agent
    ) -> tuple[Agent, Curriculum, List[Task]]:
        """Resume curriculum from saved state."""
        try:
            # Reconstruct curriculum (simplified for demo)
            from .models.system import CurriculumStrategy, DifficultyLevel

            curriculum_data = saved_state["curriculum"]
            curriculum = Curriculum(
                id=curriculum_data["id"],
                name=curriculum_data["name"],
                agent_id=curriculum_data["agent_id"],
                strategy=CurriculumStrategy(curriculum_data["strategy"]),
                current_difficulty=DifficultyLevel(
                    curriculum_data["current_difficulty"]
                ),
                target_skills=curriculum_data["target_skills"],
                completed_skills=curriculum_data["completed_skills"],
                active_tasks=curriculum_data["active_tasks"],
                task_queue=curriculum_data["task_queue"],
                completed_tasks=curriculum_data["completed_tasks"],
                failed_tasks=curriculum_data["failed_tasks"],
                progress_metrics={
                    k: Decimal(str(v))
                    for k, v in curriculum_data["progress_metrics"].items()
                },
                generation_parameters=curriculum_data["generation_parameters"],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

            # Reconstruct active tasks (simplified)
            active_tasks = []
            for task_data in saved_state.get("active_tasks", []):
                task = self._deserialize_task(task_data)
                active_tasks.append(task)

            self.logger.info(
                f"Resumed curriculum with {len(active_tasks)} active tasks"
            )
            return current_agent, curriculum, active_tasks

        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"Failed to resume from state: {e}")
            # Fall back to new curriculum
            return self._initialize_new_curriculum(current_agent, None)

    def _initialize_new_curriculum(
        self, agent: Agent, environment: Optional[Environment]
    ) -> tuple[Agent, Curriculum, List[Task]]:
        """Initialize a new curriculum for the agent."""
        from .models.system import CurriculumStrategy, DifficultyLevel

        curriculum = Curriculum(
            id=f"curriculum_{agent.id}",
            name=f"VOYAGER Curriculum for {agent.name}",
            agent_id=agent.id,
            strategy=CurriculumStrategy.PROGRESSIVE,
            current_difficulty=DifficultyLevel.BEGINNER,
            target_skills=[
                "basic_trading",
                "risk_management",
                "market_analysis",
                "strategy_development",
                "portfolio_management",
            ],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        return agent, curriculum, []

    def _deserialize_task(self, task_data: Dict[str, Any]) -> Task:
        """Deserialize task from saved data."""
        from .models.system import DifficultyLevel, TaskPriority, TaskStatus, TaskType

        return Task(
            id=task_data["id"],
            title=task_data["title"],
            description=task_data["description"],
            task_type=TaskType(task_data["task_type"]),
            priority=TaskPriority(task_data["priority"]),
            difficulty=DifficultyLevel(task_data["difficulty"]),
            status=TaskStatus(task_data["status"]),
            objectives=task_data["objectives"],
            success_criteria=task_data["success_criteria"],
            required_skills=task_data["required_skills"],
            parameters=task_data["parameters"],
            progress_percentage=Decimal(str(task_data["progress_percentage"])),
            created_at=(
                datetime.fromisoformat(task_data["created_at"])
                if task_data["created_at"]
                else None
            ),
            started_at=(
                datetime.fromisoformat(task_data["started_at"])
                if task_data["started_at"]
                else None
            ),
            completed_at=(
                datetime.fromisoformat(task_data["completed_at"])
                if task_data["completed_at"]
                else None
            ),
            updated_at=datetime.utcnow(),
        )


# Factory function for easy instantiation
def create_curriculum_service(
    config: Dict[str, Any] = None,
) -> ResumableCurriculumService:
    """
    Factory function to create a fully configured curriculum service.

    Args:
        config: Configuration dictionary

    Returns:
        Configured ResumableCurriculumService instance
    """
    default_config = {
        "auto_save": True,
        "save_interval_tasks": 1,
        "storage_path": "curriculum_data",
        "log_level": "INFO",
    }

    if config:
        default_config.update(config)

    return ResumableCurriculumService(default_config, default_config["storage_path"])
