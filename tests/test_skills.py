"""Tests for the SkillLibrary functionality."""

import tempfile
from pathlib import Path

from voyager_trader.core import TradingConfig
from voyager_trader.skills import SkillLibrary, TradingSkill


class TestTradingSkill:
    """Test the TradingSkill dataclass."""

    def test_skill_creation(self):
        """Test creating a trading skill."""
        skill = TradingSkill(
            name="trend_following",
            description="Basic trend following strategy",
            code="def strategy(): pass",
            performance_metrics={"return": 0.05},
        )

        assert skill.name == "trend_following"
        assert skill.description == "Basic trend following strategy"
        assert skill.code == "def strategy(): pass"
        assert skill.performance_metrics == {"return": 0.05}
        assert skill.usage_count == 0
        assert skill.success_rate == 0.0
        assert skill.prerequisites == []
        assert skill.tags == []


class TestSkillLibrary:
    """Test the SkillLibrary class."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = TradingConfig()
        self.config.skill_library_path = self.temp_dir

    def test_initialization(self):
        """Test skill library initialization."""
        library = SkillLibrary(self.config)

        assert library.config == self.config
        assert library.skills == {}
        assert library.skill_dependencies == {}
        assert Path(self.temp_dir).exists()

    def test_add_skill(self):
        """Test adding a skill to the library."""
        library = SkillLibrary(self.config)
        skill = TradingSkill(
            name="test_skill",
            description="Test skill",
            code="def test(): pass",
            performance_metrics={},
        )

        library.add_skill(skill)

        assert "test_skill" in library.skills
        assert library.skills["test_skill"] == skill

    def test_get_skill(self):
        """Test retrieving a skill."""
        library = SkillLibrary(self.config)
        skill = TradingSkill(
            name="test_skill",
            description="Test skill",
            code="def test(): pass",
            performance_metrics={},
        )
        library.add_skill(skill)

        retrieved = library.get_skill("test_skill")
        assert retrieved == skill

        not_found = library.get_skill("nonexistent")
        assert not_found is None

    def test_search_skills_by_success_rate(self):
        """Test searching skills by success rate."""
        library = SkillLibrary(self.config)

        skill1 = TradingSkill("skill1", "desc", "code", {})
        skill1.success_rate = 0.8
        skill2 = TradingSkill("skill2", "desc", "code", {})
        skill2.success_rate = 0.5

        library.add_skill(skill1)
        library.add_skill(skill2)

        results = library.search_skills(min_success_rate=0.7)
        assert len(results) == 1
        assert results[0] == skill1

    def test_search_skills_by_tags(self):
        """Test searching skills by tags."""
        library = SkillLibrary(self.config)

        skill1 = TradingSkill("skill1", "desc", "code", {}, tags=["momentum"])
        skill2 = TradingSkill("skill2", "desc", "code", {}, tags=["mean_reversion"])

        library.add_skill(skill1)
        library.add_skill(skill2)

        results = library.search_skills(tags=["momentum"])
        assert len(results) == 1
        assert results[0] == skill1

    def test_compose_skills(self):
        """Test composing multiple skills."""
        library = SkillLibrary(self.config)

        skill1 = TradingSkill("skill1", "desc", "def func1(): pass", {})
        skill2 = TradingSkill("skill2", "desc", "def func2(): pass", {})

        library.add_skill(skill1)
        library.add_skill(skill2)

        composed = library.compose_skills(["skill1", "skill2"])
        assert "def func1(): pass" in composed
        assert "def func2(): pass" in composed

    def test_compose_skills_missing(self):
        """Test composing with missing skills."""
        library = SkillLibrary(self.config)

        composed = library.compose_skills(["nonexistent"])
        assert composed is None

    def test_update_skill_performance(self):
        """Test updating skill performance."""
        library = SkillLibrary(self.config)

        skill = TradingSkill("test_skill", "desc", "code", {})
        library.add_skill(skill)

        performance = {"success": 1.0, "return": 0.05}
        library.update_skill_performance("test_skill", performance)

        updated_skill = library.get_skill("test_skill")
        assert updated_skill.usage_count == 1
        assert updated_skill.success_rate == 1.0
        assert "return" in updated_skill.performance_metrics
