# VOYAGER-Trader

An autonomous, self-improving trading system inspired by the [VOYAGER project](https://voyager.minedojo.org/) that focuses on discovering and developing trading knowledge through systematic exploration.

## Overview

VOYAGER-Trader implements the three key components from the original VOYAGER system, adapted for financial markets:

1. **🎯 Automatic Curriculum**: Generates progressive trading tasks and challenges based on market conditions and agent performance
2. **📚 Skill Library**: Stores and manages learned trading strategies with composition and reuse capabilities  
3. **🔄 Iterative Prompting**: Uses LLMs to generate and refine trading strategies through iterative feedback

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-org/voyager-trader.git
cd voyager-trader
```

2. Set up virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install pre-commit hooks (recommended for development):
```bash
pre-commit install
```

## Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Test specific component
python -m pytest tests/test_core.py -v
```

### Code Quality

The project enforces code quality through automated checks:

```bash
# Format code
black src/ tests/

# Sort imports  
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Development Workflow

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes following [conventional commits](https://www.conventionalcommits.org/)
3. Run tests: `pytest tests/`
4. Create pull request

## Architecture

The system follows SOLID principles and uses Architecture Decision Records (ADRs) for major design decisions.

### Core Components

- **`VoyagerTrader`**: Main system orchestrator
- **`AutomaticCurriculum`**: Task generation and progression
- **`SkillLibrary`**: Strategy storage and composition
- **`IterativePrompting`**: LLM-based strategy development

### Key Design Principles

- **Test-Driven Development**: Write tests before implementation
- **Issue-Driven**: Use GitHub issues for task management
- **Conventional Commits**: Structured commit messages
- **Automated Quality Gates**: All PRs must pass linting, testing, and coverage

## Documentation

- [Goals and Objectives](goals.md)
- [Technical Design Document](VOYAGER-Trader_PDR_v3.md)
- [Architecture Decision Records](docs/adr/)
- [Development Guidelines](CLAUDE.md)

## Contributing

1. Check [open issues](https://github.com/your-org/voyager-trader/issues)
2. Review [contribution guidelines](CONTRIBUTING.md)
3. Follow the development workflow above
4. Ensure all quality gates pass

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by the [VOYAGER project](https://github.com/MineDojo/Voyager) from MineDojo
- Built following modern software engineering best practices
- Designed for autonomous exploration and continuous learning
