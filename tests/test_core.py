"""Tests for the core VoyagerTrader functionality."""

# Core tests - no external mocks needed currently

from voyager_trader.core import TradingConfig, VoyagerTrader


class TestTradingConfig:
    """Test the TradingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TradingConfig()
        assert config.model_name == "gpt-4"
        assert config.max_iterations == 1000
        assert config.skill_library_path == "skills/"
        assert config.curriculum_temperature == 0.1
        assert config.enable_risk_management is True
        assert config.max_position_size == 0.1


class TestVoyagerTrader:
    """Test the main VoyagerTrader class."""

    def test_initialization(self):
        """Test VoyagerTrader initialization."""
        trader = VoyagerTrader()
        assert trader.config is not None
        assert trader.curriculum is not None
        assert trader.skill_library is not None
        assert trader.prompting is not None
        assert trader.is_running is False
        assert trader.current_task is None

    def test_initialization_with_config(self):
        """Test VoyagerTrader initialization with custom config."""
        config = TradingConfig(model_name="gpt-3.5-turbo", max_iterations=500)
        trader = VoyagerTrader(config)
        assert trader.config.model_name == "gpt-3.5-turbo"
        assert trader.config.max_iterations == 500

    def test_start_system(self):
        """Test starting the trading system."""
        trader = VoyagerTrader()
        trader.start()
        assert trader.is_running is True

    def test_stop_system(self):
        """Test stopping the trading system."""
        trader = VoyagerTrader()
        trader.start()
        trader.stop()
        assert trader.is_running is False

    def test_get_status(self):
        """Test getting system status."""
        trader = VoyagerTrader()
        status = trader.get_status()

        assert "is_running" in status
        assert "current_task" in status
        assert "skills_learned" in status
        assert "performance" in status
        assert status["is_running"] is False
        assert status["current_task"] is None
