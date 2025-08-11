# VOYAGER-Trader Makefile
# Automation for installation, configuration, testing, and running the system

.PHONY: help install configure config-check config-validate setup-env test test-unit test-integration test-components \
		test-curriculum test-skills test-prompting test-core \
		run clean format lint typecheck quality-check \
		setup-dev docs coverage get-system-info get-hardware-info \
		get-network-info get-disk-info get-process-info get-env-info \
		get-project-data get-all-data admin-setup admin-test admin-start admin-backend admin-frontend \
		ps-show ps-stop ps-start

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
configure: venv-check ## Interactive configuration setup with guided API key entry
	@echo "🚀 Starting VOYAGER-Trader interactive configuration..."
	@$(PYTHON_VENV) scripts/interactive_config.py
	@echo "🎯 Creating required directories..."
	@mkdir -p skills
	@mkdir -p curriculum_data
	@mkdir -p logs

config-check: venv-check ## Check configuration status without setup
	@echo "🔍 Checking VOYAGER-Trader configuration status..."
	@$(PYTHON_VENV) scripts/validate_config.py

setup-env: ## Create environment variables template file
	@echo "🔧 Creating environment variables template..."
	@if [ ! -f ".env" ]; then \
		cp scripts/setup_env_template.sh .env; \
		echo "✅ Created .env file from template"; \
		echo "📝 Edit .env with your API keys, then run: source .env"; \
		echo "💡 After setting up, run: make config-check"; \
		echo "🚀 Or use interactive setup: make configure"; \
	else \
		echo "⚠️  .env file already exists. Template available at scripts/setup_env_template.sh"; \
	fi

config-validate: venv-check ## Quick validation check (used internally)
	@$(PYTHON_VENV) scripts/validate_config.py >/dev/null 2>&1

# 3. Testing
test: venv-check clean ## Run all tests with coverage
	@echo "🧪 Running complete test suite..."
	@$(PYTEST) tests/ -v --cov=src --cov-report=term-missing --cov-report=html --cov-report=xml
	@echo "📊 Coverage report generated in htmlcov/"

test-unit: venv-check ## Run unit tests only
	@echo "🔬 Running unit tests..."
	@$(PYTEST) tests/ -v -m "unit or not (integration or slow)" --no-cov

test-no-cov: venv-check ## Run all tests without coverage
	@echo "🧪 Running all tests (no coverage)..."
	@$(PYTEST) tests/ -v --no-cov

test-integration: venv-check ## Run integration tests only
	@echo "🔗 Running integration tests..."
	@$(PYTEST) tests/ -v -m "integration"

test-fast: venv-check ## Run fast tests (exclude slow tests)
	@echo "⚡ Running fast tests..."
	@$(PYTEST) tests/ -v -m "not slow" --no-cov

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
run: venv-check ## Start the VOYAGER-Trader system
	@echo "🔍 Checking configuration..."
	@if ! $(MAKE) config-validate; then \
		echo "❌ System not configured. Run 'make configure' first."; \
		exit 1; \
	fi
	@echo "🚀 Starting VOYAGER-Trader system..."
	@echo "⚠️  Note: This will start the autonomous trading system"
	@echo "Press Ctrl+C to stop"
	@$(PYTHON_VENV) -c "from src.voyager_trader import VoyagerTrader; import time; trader = VoyagerTrader(); trader.start(); print('✅ System started - monitoring...'); time.sleep(3600)"

run-demo: venv-check ## Run system in demo mode (safe)
	@echo "🔍 Checking configuration..."
	@if ! $(MAKE) config-validate; then \
		echo "❌ System not configured. Run 'make configure' first."; \
		exit 1; \
	fi
	@echo "🎭 Running VOYAGER-Trader in demo mode..."
	@$(PYTHON_VENV) -c "from src.voyager_trader import VoyagerTrader; from src.voyager_trader.core import TradingConfig; config = TradingConfig(max_iterations=10); trader = VoyagerTrader(config); trader.start(); print('✅ Demo completed - system started and stopped successfully'); trader.stop()"

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
clean-coverage: ## Clean coverage data only
	@echo "🧹 Cleaning coverage data..."
	@rm -rf .coverage*
	@rm -rf htmlcov/
	@rm -rf coverage.xml
	@rm -rf coverage.json

clean: clean-coverage ## Clean up generated files and caches
	@echo "🧹 Cleaning up..."
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
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
	@echo ""
	@echo "📋 Configuration Status:"
	@$(PYTHON_VENV) scripts/validate_config.py
	@echo ""

# # 11. System data retrieval
# get-system-info: ## Display comprehensive system information
# 	@echo "System Information"
# 	@echo "=================="
# 	@echo "Operating System: $$(uname -s)"
# 	@echo "Architecture: $$(uname -m)"
# 	@echo "Kernel Version: $$(uname -r)"
# 	@echo "Hostname: $$(hostname)"
# 	@echo "Current User: $$(whoami)"
# 	@echo "Working Directory: $$(pwd)"
# 	@echo "Date/Time: $$(date)"
# 	@echo ""

# get-hardware-info: ## Display hardware information
# 	@echo "Hardware Information"
# 	@echo "==================="
# 	@if command -v sw_vers >/dev/null 2>&1; then \
# 		echo "macOS Version: $$(sw_vers -productVersion)"; \
# 		echo "Build Version: $$(sw_vers -buildVersion)"; \
# 		echo "CPU Info: $$(sysctl -n machdep.cpu.brand_string)"; \
# 		echo "Total Memory: $$(sysctl -n hw.memsize | awk '{printf "%.1f GB", $$1/1024/1024/1024}')"; \
# 		echo "CPU Cores: $$(sysctl -n hw.ncpu)"; \
# 	elif command -v lscpu >/dev/null 2>&1; then \
# 		echo "CPU Info:"; lscpu | grep -E "(Model name|CPU\(s\)|Thread|Core|Socket)"; \
# 		echo "Memory Info:"; free -h 2>/dev/null || echo "Memory info not available"; \
# 	else \
# 		echo "Hardware info commands not available"; \
# 	fi
# 	@echo ""

# get-network-info: ## Display network configuration
# 	@echo "Network Information"
# 	@echo "==================="
# 	@echo "Network Interfaces:"
# 	@if command -v ifconfig >/dev/null 2>&1; then \
# 		ifconfig | grep -E "(^[a-z]|inet )" | grep -v "127.0.0.1" | head -10; \
# 	elif command -v ip >/dev/null 2>&1; then \
# 		ip addr show | grep -E "(^[0-9]|inet )" | head -10; \
# 	else \
# 		echo "Network info commands not available"; \
# 	fi
# 	@echo ""

# get-disk-info: ## Display disk usage information
# 	@echo "Disk Usage Information"
# 	@echo "======================"
# 	@echo "Filesystem Usage:"
# 	@df -h | head -10
# 	@echo ""
# 	@echo "Project Directory Usage:"
# 	@du -sh . 2>/dev/null || echo "Cannot determine directory size"
# 	@echo ""

# get-process-info: ## Display process and resource information
# 	@echo "Process Information"
# 	@echo "==================="
# 	@echo "Load Average: $$(uptime | awk -F'load average:' '{print $$2}')"
# 	@echo ""
# 	@echo "Top CPU Processes:"
# 	@if command -v top >/dev/null 2>&1; then \
# 		top -l 1 -n 5 -stats pid,command,cpu 2>/dev/null | tail -n +12 | head -5 || ps aux --sort=-%cpu | head -6; \
# 	else \
# 		ps aux --sort=-%cpu 2>/dev/null | head -6 || echo "Process info not available"; \
# 	fi
# 	@echo ""

# get-env-info: ## Display environment and development information
# 	@echo "Environment Information"
# 	@echo "======================="
# 	@echo "Shell: $$SHELL"
# 	@echo "PATH (first 3 entries): $$(echo $$PATH | tr ':' '\n' | head -3 | tr '\n' ':' | sed 's/:$$//')"
# 	@echo "Python Locations:"
# 	@which python3 2>/dev/null || echo "  python3: not found"
# 	@which python 2>/dev/null || echo "  python: not found"
# 	@if command -v git >/dev/null 2>&1; then \
# 		echo "Git Version: $$(git --version)"; \
# 		echo "Git Config User: $$(git config --get user.name 2>/dev/null || echo 'Not configured')"; \
# 		echo "Git Config Email: $$(git config --get user.email 2>/dev/null || echo 'Not configured')"; \
# 	fi
# 	@echo ""

# get-project-data: venv-check ## Display project-specific data and metrics
# 	@echo "Project Data & Metrics"
# 	@echo "======================"
# 	@echo "Repository Status:"
# 	@if command -v git >/dev/null 2>&1; then \
# 		echo "  Branch: $$(git branch --show-current 2>/dev/null || echo 'Not a git repository')"; \
# 		echo "  Last Commit: $$(git log -1 --format='%h - %s (%cr)' 2>/dev/null || echo 'No commits')"; \
# 		echo "  Modified Files: $$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"; \
# 	fi
# 	@echo ""
# 	@echo "Project Structure:"
# 	@find . -maxdepth 3 -name "*.py" | wc -l | awk '{print "  Python Files: " $$1}'
# 	@find . -maxdepth 2 -name "test_*.py" -o -name "*_test.py" | wc -l | awk '{print "  Test Files: " $$1}'
# 	@echo "  Project Size: $$(du -sh . | cut -f1)"
# 	@echo ""
# 	@if [ -f "pyproject.toml" ]; then \
# 		echo "Dependencies (from pyproject.toml):"; \
# 		grep -A 20 "\[project\]" pyproject.toml | grep -E "dependencies|version" | head -5; \
# 	fi
# 	@echo ""

# get-all-data: get-system-info get-hardware-info get-network-info get-disk-info get-process-info get-env-info get-project-data ## Get all system data in one command
# 	@echo "🎯 Complete system data collection finished!"
# 	@echo ""

# Quick start message
# 12. Admin Interface
admin-setup: venv-check ## Install admin interface dependencies
	@echo "🌐 Setting up admin interface..."
	@echo "📦 Installing FastAPI dependencies..."
	@$(PIP) install fastapi uvicorn[standard] python-multipart python-jose[cryptography] passlib[bcrypt]
	@echo "📦 Installing React dependencies..."
	@cd admin-ui && npm install
	@echo "✅ Admin interface dependencies installed!"

admin-test: venv-check ## Test admin interface setup
	@echo "🔍 Testing admin interface setup..."
	@$(PYTHON_VENV) test_admin_setup.py

admin-start: venv-check ## Start admin interface (backend + frontend instructions)
	@echo "🚀 Starting admin interface..."
	@$(PYTHON_VENV) start_admin.py

admin-backend: venv-check ## Start only the admin backend API server
	@echo "🔧 Starting admin backend API server..."
	@$(PYTHON_VENV) -m uvicorn src.voyager_trader.admin_api:app --reload --host 0.0.0.0 --port 8001

admin-frontend: ## Start only the React frontend (requires backend running)
	@echo "🎨 Starting React frontend..."
	@echo "⚠️  Make sure backend is running (make admin-backend)"
	@cd admin-ui && npm start

# JWT Secret Management
jwt-generate: venv-check ## Generate a new JWT secret for admin interface
	@echo "🔐 Generating JWT secret..."
	@$(PYTHON_VENV) scripts/generate_jwt_secret.py

jwt-setup: venv-check ## Generate and save JWT secret to .env file
	@echo "🔐 Setting up persistent JWT secret..."
	@$(PYTHON_VENV) scripts/generate_jwt_secret.py --env

# Process Management
ps-show: ## Show all VOYAGER-Trader related processes
	@echo "🔍 VOYAGER-Trader Processes"
	@echo "==========================="
	@echo ""
	@echo "Backend API (uvicorn):"
	@ps aux | grep -E "(uvicorn.*admin_api|uvicorn.*voyager)" | grep -v grep || echo "  ❌ No backend processes found"
	@echo ""
	@echo "React Frontend (node/npm):"
	@ps aux | grep -E "(react-scripts|admin-ui)" | grep -v grep || echo "  ❌ No frontend processes found"
	@echo ""
	@echo "Python processes:"
	@ps aux | grep -E "(python.*voyager|start_admin)" | grep -v grep || echo "  ❌ No Python VOYAGER processes found"
	@echo ""
	@echo "Port usage:"
	@echo "  Backend API (8001):"
	@lsof -ti :8001 >/dev/null 2>&1 && echo "  ✅ Port 8001 in use" || echo "  ❌ Port 8001 free"
	@echo "  Frontend (3001):"
	@lsof -ti :3001 >/dev/null 2>&1 && echo "  ✅ Port 3001 in use" || echo "  ❌ Port 3001 free"

ps-stop: ## Stop all VOYAGER-Trader processes
	@echo "🛑 Stopping all VOYAGER-Trader processes..."
	@echo ""
	@echo "Stopping backend API servers..."
	@pkill -f "uvicorn.*admin_api" 2>/dev/null && echo "  ✅ Stopped admin API server" || echo "  ℹ️  No admin API server running"
	@pkill -f "uvicorn.*voyager" 2>/dev/null && echo "  ✅ Stopped VOYAGER servers" || echo "  ℹ️  No VOYAGER servers running"
	@echo ""
	@echo "Stopping React frontend..."
	@pkill -f "react-scripts" 2>/dev/null && echo "  ✅ Stopped React frontend" || echo "  ℹ️  No React frontend running"
	@pkill -f "admin-ui.*npm" 2>/dev/null && echo "  ✅ Stopped npm processes" || echo "  ℹ️  No npm processes running"
	@echo ""
	@echo "Stopping Python processes..."
	@pkill -f "python.*start_admin" 2>/dev/null && echo "  ✅ Stopped start_admin processes" || echo "  ℹ️  No start_admin processes running"
	@echo ""
	@echo "Killing processes on reserved ports..."
	@lsof -ti :8001 | xargs kill 2>/dev/null && echo "  ✅ Freed port 8001" || echo "  ℹ️  Port 8001 already free"
	@lsof -ti :3001 | xargs kill 2>/dev/null && echo "  ✅ Freed port 3001" || echo "  ℹ️  Port 3001 already free"
	@echo ""
	@echo "🎉 All processes stopped!"

ps-start: venv-check ## Start all VOYAGER-Trader processes (backend + frontend)
	@echo "🚀 Starting all VOYAGER-Trader processes..."
	@echo ""
	@echo "Starting backend API server on port 8001..."
	@nohup $(PYTHON_VENV) -m uvicorn src.voyager_trader.admin_api:app --reload --host 0.0.0.0 --port 8001 > logs/backend.log 2>&1 &
	@echo "  ✅ Backend starting in background (logs: logs/backend.log)"
	@sleep 3
	@echo ""
	@echo "Checking backend health..."
	@curl -s http://localhost:8001/api/health > /dev/null && echo "  ✅ Backend is healthy" || echo "  ⚠️  Backend may still be starting..."
	@echo ""
	@echo "Starting React frontend on port 3001..."
	@echo "  ℹ️  Starting React development server..."
	@echo "  ℹ️  This may take a moment to compile and start..."
	@cd admin-ui && nohup npm start > ../logs/frontend.log 2>&1 &
	@echo "  ✅ Frontend starting in background (logs: logs/frontend.log)"
	@sleep 5
	@echo ""
	@echo "🎉 All processes started!"
	@echo ""
	@echo "Process status:"
	@ps aux | grep -E "(uvicorn.*admin_api)" | grep -v grep | wc -l | xargs -I {} echo "  Backend: {} process(es) running"
	@ps aux | grep -E "(react-scripts)" | grep -v grep | wc -l | xargs -I {} echo "  Frontend: {} process(es) running"
	@echo ""
	@echo "Access points:"
	@echo "  🌐 Admin Interface: http://localhost:3001"
	@echo "  🔧 API Documentation: http://localhost:8001/docs"
	@echo "  ❤️  Health Check: http://localhost:8001/api/health"
	@echo ""
	@echo "Credentials: admin / admin123"
	@echo ""
	@echo "Commands:"
	@echo "  make ps-stop      # Stop all processes"
	@echo "  make ps-show      # Show detailed process status"

welcome: ## Show welcome message and quick start guide
	@echo "🚀 Welcome to VOYAGER-Trader!"
	@echo "=============================="
	@echo ""
	@echo "An autonomous, self-improving trading system inspired by VOYAGER"
	@echo ""
	@echo "Quick Start:"
	@echo "  make install      # Set up everything"
	@echo "  make test         # Run tests"
	@echo "  make run-demo     # Try it out safely"
	@echo "  make admin-setup  # Set up admin interface"
	@echo "  make ps-start     # Start all processes (backend + frontend)"
	@echo ""
	@echo "Process Management:"
	@echo "  make ps-start     # Start all processes"
	@echo "  make ps-stop      # Stop all processes"
	@echo "  make ps-show      # Show process status"
	@echo ""
	@echo "Development:"
	@echo "  make help         # See all commands"
	@echo "  make status       # Check system status"
	@echo ""
