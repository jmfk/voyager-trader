# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

VOYAGER-Trader: An autonomous, self-improving trading system inspired by the VOYAGER project that focuses on discovering and developing trading knowledge through systematic exploration. Like VOYAGER's approach to open-ended skill learning in Minecraft, our system will continuously explore financial markets, acquire diverse trading skills, and make novel discoveries without human intervention.

## Project Documentation

- **goals.md** - Core project objectives and success metrics, incorporating VOYAGER's three key components: automatic curriculum, skill library, and iterative prompting
- **VOYAGER-Trader_PDR_v2.md** - Technical design document
- **Research Library** - Collection of relevant academic papers including the original VOYAGER paper and related work on autonomous agents, lifelong learning, and algorithmic strategy development

## Development Commands

### Core Development Workflow
```bash
# Setup virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Test-driven development cycle
python -m pytest tests/ -v              # Run tests first (should fail initially)
python -m pytest tests/ --cov=src       # Run with coverage

# Code quality checks
flake8 src/ tests/                       # Linting
black src/ tests/                        # Code formatting
mypy src/                               # Type checking

# GitHub integration
gh issue list --state open              # Check current issues
gh issue create --title "Feature: X"    # Create new issue
git checkout -b feature/issue-123-name  # Create feature branch
git commit -m "feat: implement X (refs #123)"  # Commit with issue reference
gh pr create --title "feat: implement X" --body "Closes #123"  # Create PR
```

### LLM Service Setup
```bash
# Install and setup Ollama for local models
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve                            # Start Ollama server
ollama pull llama2                      # Pull Llama2 model
ollama pull codellama                   # Pull CodeLlama model
ollama pull mistral                     # Pull Mistral model

# Test LLM service
python examples/llm_service_examples.py # Run example usage

# Set up API keys (required for remote providers)
export OPENAI_API_KEY="sk-..."         # OpenAI API key
export ANTHROPIC_API_KEY="sk-ant-..."  # Anthropic Claude API key
```

### ADR (Architecture Decision Record) Commands
```bash
# Check existing architectural decisions
ls docs/adr/
cat docs/adr/README.md

# Create new ADR
NEXT_NUM=$(printf "%04d" $(($(ls docs/adr/[0-9]*.md | tail -1 | grep -o '[0-9]\{4\}') + 1)))
cp docs/adr/0000-template.md docs/adr/${NEXT_NUM}-your-decision.md

# Validate ADR format (automatic in PR)
grep -q "# Status\|# Context\|# Decision\|# Consequences" docs/adr/*.md
```

### Development Principles
- **Test-First Development**: Write tests before implementation
- **Issue-Driven**: Use GitHub issues for task management
- **ADR Documentation**: Document all architectural decisions
- **Conventional Commits**: Use conventional commit format with issue references
- **State-of-the-Art Design**: Follow SOLID, DRY, KISS, YAGNI principles
- **Automated Quality Gates**: All PRs must pass linting, testing, and coverage requirements


## Architecture Overview

**All architectural decisions are documented in ADRs (Architecture Decision Records)**

See `docs/adr/` for detailed architectural decisions including:
- Framework choice and rationale (ADR required)
- State management approach (ADR required)  
- API design patterns (ADR required)
- Directory structure conventions (ADR required)
- Data flow patterns (ADR required)
- Database design decisions (ADR required)
- Integration patterns (ADR required)

Current ADRs:
- [ADR-0001: Record Architecture Decisions](docs/adr/0001-record-architecture-decisions.md)
- [ADR-0014: Iterative Prompting Architecture](docs/adr/0014-iterative-prompting-architecture.md)
- [ADR-0015: Centralized LLM Service](docs/adr/0015-centralized-llm-service.md)

**Key Principle**: Any architectural decision affecting system design, technology choices, or development patterns MUST be documented as an ADR before implementation.

## Development Notes

### LLM Service Integration

**Centralized LLM Service**: All AI/LLM interactions in the system go through the centralized LLM service (`src/voyager_trader/llm_service.py`). This provides:

- **Single Integration Point**: All components use the same LLM interface
- **Multi-Provider Support**: OpenAI, Anthropic Claude, Ollama (local models)
- **Automatic Fallback**: Failed requests automatically retry with fallback providers
- **OpenAI-Compatible API**: Universal client that works with existing OpenAI code
- **Configuration-Driven**: Easy to enable/disable providers and models

**Key Integration Points:**
- Iterative Prompting System (`src/voyager_trader/prompting.py`)
- Trading Strategy Generation
- Market Analysis and Research
- Code Generation and Evaluation

**Usage Examples:**
```python
from src.voyager_trader.llm_service import chat_completion_create

# Simple usage (automatic provider selection)
response = await chat_completion_create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Analyze AAPL"}]
)

# Force specific provider (e.g., local Ollama)
response = await chat_completion_create(
    model="llama2",
    messages=[{"role": "user", "content": "Generate strategy"}],
    provider="ollama"
)
```

**Documentation:**
- [LLM Service Usage Guide](docs/llm-service-usage.md)
- [Configuration Examples](config/llm_service_example.yaml)
- [Code Examples](examples/llm_service_examples.py)
