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

**Key Principle**: Any architectural decision affecting system design, technology choices, or development patterns MUST be documented as an ADR before implementation.

## Development Notes

*Project-specific development guidelines will be added here*

This might include:
- Code style preferences
- Testing strategies
- Deployment processes
- Environment setup requirements