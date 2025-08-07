# VOYAGER-Trader Makefile
# Automation for installation, configuration, testing, and running the system

.PHONY: help install configure test test-unit test-integration test-components \
		test-curriculum test-skills test-prompting test-core \
		run clean format lint typecheck quality-check \
		setup-dev docs coverage

# Default target
help: ## Show this help message
	@echo "VOYAGER-Trader Development Automation"
	@echo "====================================="
	@echo ""
	@echo "Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# Variables
PYTHON := python3.12
VENV_DIR := venv
PIP := $(VENV_DIR)/bin/pip
PYTHON_VENV := $(VENV_DIR)/bin/python
PYTEST := $(VENV_DIR)/bin/pytest
BLACK := $(VENV_DIR)/bin/black
FLAKE8 := $(VENV_DIR)/bin/flake8
MYPY := $(VENV_DIR)/bin/mypy
ISORT := $(VENV_DIR)/bin/isort
PRECOMMIT := $(VENV_DIR)/bin/pre-commit

# Check if virtual environment exists
venv-check:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "❌ Virtual environment not found. Run 'make install' first."; \
		exit 1; \
	fi

# 1. Installation and Setup
install: ## Install all dependencies and set up the development environment
	@echo "🚀 Installing VOYAGER-Trader development environment..."
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "📦 Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	@echo "⬆️  Upgrading pip..."
	@$(PIP) install --upgrade pip setuptools wheel
	@echo "📋 Installing requirements..."
	@$(PIP) install -r requirements.txt
	@echo "🛠️  Installing development dependencies..."
	@$(PIP) install -e .[dev]
	@echo "🔧 Installing pre-commit hooks..."
	@$(PRECOMMIT) install
	@echo "✅ Installation complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  make configure  # Configure the system"
	@echo "  make test       # Run tests"
	@echo "  make run        # Start the system"

# 2. Configuration
configure: venv-check ## Configure and validate system settings
	@echo "⚙️  Configuring VOYAGER-Trader system..."
	@echo "🔍 Validating Python version..."
	@$(PYTHON_VENV) -c "import sys; assert sys.version_info >= (3, 12), f'Python 3.12+ required, got {sys.version_info}'"
	@echo "📦 Verifying core dependencies..."
	@$(PYTHON_VENV) -c "import numpy, pandas, openai, pytest; print('✅ Core dependencies verified')"
	@echo "🏗️  Validating project structure..."
	@test -d src/voyager_trader || (echo "❌ Source directory missing"; exit 1)
	@test -d tests || (echo "❌ Tests directory missing"; exit 1)
	@test -f requirements.txt || (echo "❌ Requirements file missing"; exit 1)
	@echo "🎯 Creating skills directory if needed..."
	@mkdir -p skills
	@mkdir -p curriculum_data
	@echo "✅ Configuration complete!"

# 3. Testing
test: venv-check ## Run all tests with coverage
	@echo "🧪 Running complete test suite..."
	@$(PYTEST) tests/ -v --cov=src --cov-report=term-missing --cov-report=html --cov-report=xml
	@echo "📊 Coverage report generated in htmlcov/"

test-unit: venv-check ## Run unit tests only
	@echo "🔬 Running unit tests..."
	@$(PYTEST) tests/ -v -m "unit or not (integration or slow)"

test-integration: venv-check ## Run integration tests only
	@echo "🔗 Running integration tests..."
	@$(PYTEST) tests/ -v -m "integration"

test-fast: venv-check ## Run fast tests (exclude slow tests)
	@echo "⚡ Running fast tests..."
	@$(PYTEST) tests/ -v -m "not slow"

# 4. Component-specific testing
test-components: test-curriculum test-skills test-prompting test-core ## Run all component tests

test-curriculum: venv-check ## Test automatic curriculum component
	@echo "📚 Testing Automatic Curriculum..."
	@$(PYTEST) tests/test_curriculum*.py -v

test-skills: venv-check ## Test skill library component
	@echo "🎯 Testing Skill Library..."
	@$(PYTEST) tests/test_skills*.py -v

test-prompting: venv-check ## Test iterative prompting component
	@echo "💭 Testing Iterative Prompting..."
	@$(PYTEST) tests/test_prompting*.py -v

test-core: venv-check ## Test core system
	@echo "🏗️  Testing Core System..."
	@$(PYTEST) tests/test_core*.py -v

test-models: venv-check ## Test data models
	@echo "🏛️  Testing Data Models..."
	@$(PYTEST) tests/models/ -v

# 5. System execution
run: venv-check configure ## Start the VOYAGER-Trader system
	@echo "🚀 Starting VOYAGER-Trader system..."
	@echo "⚠️  Note: This will start the autonomous trading system"
	@echo "Press Ctrl+C to stop"
	@$(PYTHON_VENV) -c "from src.voyager_trader import VoyagerTrader; import time; trader = VoyagerTrader(); trader.start(); print('✅ System started - monitoring...'); time.sleep(3600)"

run-demo: venv-check ## Run system in demo mode (safe)
	@echo "🎭 Running VOYAGER-Trader in demo mode..."
	@$(PYTHON_VENV) -c "from src.voyager_trader import VoyagerTrader, TradingConfig; config = TradingConfig(max_iterations=10); trader = VoyagerTrader(config); trader.start(); print('Demo completed'); trader.stop()"

# 6. Code quality
format: venv-check ## Format code with black and isort
	@echo "🎨 Formatting code..."
	@$(BLACK) src/ tests/
	@$(ISORT) src/ tests/
	@echo "✅ Code formatted!"

lint: venv-check ## Run linting with flake8
	@echo "🔍 Linting code..."
	@$(FLAKE8) src/ tests/

typecheck: venv-check ## Run type checking with mypy
	@echo "🔍 Type checking..."
	@$(MYPY) src/

quality-check: venv-check format lint typecheck ## Run all quality checks
	@echo "✅ All quality checks passed!"

# 7. Coverage and reporting
coverage: test ## Generate and show coverage report
	@echo "📊 Opening coverage report..."
	@if command -v open >/dev/null 2>&1; then \
		open htmlcov/index.html; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open htmlcov/index.html; \
	else \
		echo "Coverage report available at htmlcov/index.html"; \
	fi

# 8. Development utilities
clean: ## Clean up generated files and caches
	@echo "🧹 Cleaning up..."
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
	@rm -rf htmlcov/
	@rm -rf .coverage
	@rm -rf coverage.xml
	@rm -rf coverage.json
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete!"

clean-all: clean ## Clean everything including virtual environment
	@echo "🧹 Deep cleaning (including virtual environment)..."
	@rm -rf $(VENV_DIR)/
	@echo "✅ Deep cleanup complete! Run 'make install' to reinstall."

# 9. Development workflow shortcuts
dev-setup: install configure ## Complete development setup
	@echo "🎉 Development environment ready!"
	@echo ""
	@echo "Try these commands:"
	@echo "  make test       # Run tests"
	@echo "  make run-demo   # Test system"
	@echo "  make quality-check  # Check code quality"

ci-check: venv-check quality-check test ## CI/CD pipeline checks
	@echo "✅ All CI checks passed!"

# 10. Documentation and help
status: venv-check ## Show system status and configuration
	@echo "VOYAGER-Trader System Status"
	@echo "============================"
	@echo "Virtual Environment: $(VENV_DIR)"
	@echo "Python Version: $$($(PYTHON_VENV) --version)"
	@echo "Dependencies Status:"
	@$(PYTHON_VENV) -c "import sys; print(f'  Python: {sys.version}')"
	@$(PYTHON_VENV) -c "try: import numpy; print(f'  NumPy: {numpy.__version__}');\nexcept: print('  NumPy: ❌ Not installed')"
	@$(PYTHON_VENV) -c "try: import pandas; print(f'  Pandas: {pandas.__version__}');\nexcept: print('  Pandas: ❌ Not installed')"
	@$(PYTHON_VENV) -c "try: import openai; print(f'  OpenAI: {openai.__version__}');\nexcept: print('  OpenAI: ❌ Not installed')"
	@echo "Project Structure:"
	@test -d src/voyager_trader && echo "  ✅ Source code" || echo "  ❌ Source code missing"
	@test -d tests && echo "  ✅ Tests" || echo "  ❌ Tests missing"
	@test -d skills && echo "  ✅ Skills directory" || echo "  ❌ Skills directory missing"
	@echo ""

# Quick start message
welcome: ## Show welcome message and quick start guide
	@echo "🚀 Welcome to VOYAGER-Trader!"
	@echo "=============================="
	@echo ""
	@echo "An autonomous, self-improving trading system inspired by VOYAGER"
	@echo ""
	@echo "Quick Start:"
	@echo "  make install    # Set up everything"
	@echo "  make test       # Run tests"
	@echo "  make run-demo   # Try it out safely"
	@echo ""
	@echo "Development:"
	@echo "  make help       # See all commands"
	@echo "  make status     # Check system status"
	@echo ""
