# ADR-0003: Build Tool Choices

## Status

Accepted

## Context

VOYAGER-Trader requires reliable build tools for Python package development, dependency management, and deployment.

### Problem Statement

We need build tools that support modern Python packaging standards and reliable dependency management.

### Key Requirements

- Modern Python packaging standards compliance (PEP 517/518)
- Reliable dependency resolution
- Simple team onboarding
- CI/CD compatibility

## Decision

We will use setuptools + pip + pyproject.toml for build system and dependency management.

### Chosen Approach

- **setuptools**: PEP 517 compliant build backend
- **pyproject.toml**: Modern packaging metadata
- **pip**: Standard dependency installation
- **requirements.txt**: Development dependencies

### Alternative Approaches Considered

1. **Poetry**: More advanced but adds tool dependency
2. **Pipenv**: Performance issues and declining momentum
3. **Conda**: Overkill for pure Python project

## Consequences

### Positive Consequences

- Uses standard tools all Python developers know
- Maximum CI/CD compatibility
- Simple team onboarding

### Negative Consequences

- No automatic lock file generation
- Manual dependency conflict resolution

## Related

- [ADR-0001: Record Architecture Decisions](0001-record-architecture-decisions.md)
- [ADR-0002: Python Project Structure](0002-python-project-structure.md)

---

**Date**: 2025-08-06
**Author(s)**: Claude Code Assistant
**Reviewers**: Project Team
