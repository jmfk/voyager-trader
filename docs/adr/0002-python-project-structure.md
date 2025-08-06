# ADR-0002: Python Project Structure

## Status

Accepted

## Context

VOYAGER-Trader requires a well-structured Python codebase for maintainability, testability, and scalability.

### Problem Statement

We need a standardized Python project structure that supports modern development practices.

### Key Requirements

- Support for multi-component architecture
- Clear separation between source code and tests
- Modern Python packaging standards compliance
- Development tooling integration

## Decision

We will adopt a src-layout Python project structure with modern packaging using pyproject.toml.

### Chosen Approach

```
voyager-trader/
├── src/
│   └── voyager_trader/
├── tests/
├── docs/
├── pyproject.toml
└── requirements.txt
```

## Consequences

### Positive Consequences

- Clear separation prevents import issues during testing
- Modern packaging standards ensure ecosystem compatibility
- Standard structure improves developer onboarding

### Negative Consequences

- Slightly more complex than flat layout
- Requires understanding of src-layout patterns

## Related

- [ADR-0001: Record Architecture Decisions](0001-record-architecture-decisions.md)
- GitHub Issue #1: Setup: Project Foundation and Environment

---

**Date**: 2025-08-06
**Author(s)**: Claude Code Assistant
**Reviewers**: Project Team
