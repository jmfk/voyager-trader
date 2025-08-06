# ADR-0011: Python Version Standardization to 3.12

## Status

Accepted

## Context

The VOYAGER-Trader project had inconsistent Python version requirements across different configuration files and workflows, creating potential issues with CI/CD pipelines, development environment setup, and package distribution.

### Problem Statement

Multiple Python versions were specified across the codebase:
- `pyproject.toml` specified `requires-python = ">=3.10"` and mypy `python_version = "3.10"`
- GitHub Actions workflows used different Python versions (`3.11` in most workflows, `3.12` in CI)
- Package classifiers included Python 3.10, 3.11, and 3.12
- Docker containers used Python 3.11
- Test build configurations used `python_requires=">=3.10"`

This inconsistency led to:
- Unclear minimum supported Python version
- Potential CI failures due to version mismatches
- Developer confusion about which Python version to use locally
- Risk of deploying code that works in CI but fails in production

### Key Requirements

- Establish single, consistent Python version across all configurations
- Maintain compatibility with modern Python features and security updates
- Ensure CI efficiency by using a single Python version
- Support current Python ecosystem best practices

### Constraints

- Must maintain backward compatibility for existing deployments
- Should align with Python's release cycle and support timeline
- Must work with all project dependencies
- Should minimize migration effort

## Decision

We will standardize on Python 3.12 as the minimum and target Python version across the entire VOYAGER-Trader project.

### Chosen Approach

**Single Python Version Strategy**: Use Python 3.12 exclusively across all:
- Package requirements (`requires-python = ">=3.12"`)
- CI/CD workflows and Docker containers
- Development tool configurations (mypy, black)
- Package classifiers
- Test and build configurations

### Alternative Approaches Considered

1. **Multi-Version Support (3.10, 3.11, 3.12)**
   - Description: Maintain compatibility with multiple Python versions
   - Pros: Broader compatibility, easier adoption for users on older Python versions
   - Cons: Increased CI complexity, more testing overhead, potential feature limitations
   - Why rejected: Added complexity without clear benefit given our target users are expected to use modern Python

2. **Python 3.11 as Standard**
   - Description: Use Python 3.11 as the standard version
   - Pros: Slightly more conservative, still modern
   - Cons: Misses Python 3.12's performance improvements and features
   - Why rejected: Python 3.12 is stable and offers significant performance benefits

## Consequences

### Positive Consequences

- **Simplified CI/CD**: Single Python version reduces matrix complexity and CI execution time
- **Consistent Development Environment**: All developers use the same Python version
- **Modern Language Features**: Access to Python 3.12's performance improvements and new features
- **Reduced Configuration Maintenance**: Fewer version-specific configurations to maintain
- **Clear Deployment Target**: Unambiguous Python version for production environments

### Negative Consequences

- **Adoption Barrier**: Users on older Python versions must upgrade
- **Reduced Compatibility**: Cannot run on systems with only Python 3.10 or 3.11
- **Migration Effort**: Existing deployments need Python upgrade

### Neutral Consequences

- **Package Size**: No significant impact on package distribution
- **Development Workflow**: Minimal impact once environment is updated

## Implementation

### Implementation Steps

1. ✅ Update `pyproject.toml`:
   - Set `requires-python = ">=3.12"`
   - Update mypy `python_version = "3.12"`
   - Update Black `target-version = ['py312']`
   - Update package classifiers to only include Python 3.12

2. ✅ Update GitHub Actions workflows:
   - Set `PYTHON_VERSION: '3.12'` in all workflow files
   - Update `python-version` in setup-python actions
   - Update test build `python_requires` to `">=3.12"`

3. ✅ Update Docker configurations:
   - Change base image from `python:3.11-slim` to `python:3.12-slim`

4. ✅ Update release configuration:
   - Update setup.py template to use `python_requires=">=3.12"`
   - Update package classifiers in release workflow

### Success Criteria

- All CI workflows pass using Python 3.12
- Package builds successfully with Python 3.12 requirement
- No version conflicts in development or deployment environments
- Documentation accurately reflects Python 3.12 requirement

### Timeline

- Implementation: Immediate (completed in this commit)
- Migration: Affects next deployment cycle
- Full adoption: Within current development cycle

## Related

- Original issue: Python version inconsistency across project configurations
- Related to CI/CD optimization efforts
- Impacts deployment and development environment setup

## Notes

This decision aligns with Python's rapid release cycle and the project's goal of using modern, performant technology. Python 3.12 offers significant performance improvements (up to 11% faster than 3.11) and improved error messages, which benefit both development and production use.

The single-version approach prioritizes simplicity and maintainability over broad compatibility, which is appropriate for a cutting-edge trading system that benefits from the latest language improvements.

---

**Date**: 2025-01-06
**Author(s)**: Claude Code
**Reviewers**: Project Team
