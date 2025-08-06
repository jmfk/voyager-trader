# VOYAGER-Trader Automation System

This document describes the automated quality assurance and continuous integration system implemented for the VOYAGER-Trader project.

## Overview

The automation system ensures code quality through two main components:
1. **Pre-Push Hooks** - Local quality gates before code is pushed
2. **GitHub Actions** - Automated PR review and fixing

## Pre-Push Hooks

### Coverage Enforcement
- **Location**: `.pre-commit-config.yaml` (coverage-check hook)
- **Trigger**: Before every `git push`
- **Requirement**: Minimum 80% test coverage
- **Behavior**:
  - Runs test suite with coverage analysis
  - If coverage ≥ 80%: ✅ Push proceeds
  - If coverage < 80%: ❌ Push blocked with guidance

### Installation
```bash
pip install pre-commit
pre-commit install --hook-type pre-push
```

### Coverage Auto-Fix Process
When coverage is insufficient, the hook:
1. Identifies low-coverage files
2. Attempts to generate basic tests
3. Re-runs coverage check
4. Provides actionable suggestions if still failing

## GitHub Actions Workflows

### Automatic PR Review (`pr-auto-review-fix.yml`)

**Triggers:**
- PR opened/updated
- Manual workflow dispatch

**Capabilities:**
1. **Code Analysis**
   - Test coverage measurement
   - Linting (flake8)
   - Type checking (mypy)
   - Code formatting validation

2. **Auto-Fixing**
   - Applies Black code formatting
   - Fixes import sorting with isort
   - Removes unused imports
   - Generates basic tests for untested code

3. **PR Review**
   - Posts comprehensive review comments
   - Approves PRs meeting quality standards
   - Requests changes for quality issues
   - Adds relevant labels (good-coverage, needs-coverage, auto-fixed)

4. **Quality Metrics Tracking**
   - Before/after coverage comparison
   - Linting issue counts
   - Type checking status
   - Auto-fix summary

## Quality Standards

### Coverage Requirements
- **Minimum**: 80% total coverage
- **Enforcement**: Pre-push hook + CI pipeline
- **Measurement**: Line coverage using pytest-cov

### Code Quality Checks
- **Linting**: flake8 with max line length 88
- **Formatting**: Black with line length 88
- **Import sorting**: isort with Black profile
- **Type checking**: mypy for static analysis

### PR Review Criteria
- ✅ **Auto-Approve**: Coverage ≥80% + No lint/type issues
- ❌ **Request Changes**: Coverage <80% OR lint/type issues exist

## Usage Examples

### Local Development
```bash
# Normal development workflow
git add .
git commit -m "feat: new trading strategy"
git push  # Pre-push hook runs coverage check

# If coverage fails, fix tests and retry
pytest --cov=src --cov-report=html  # See detailed coverage report
# Add missing tests...
git push  # Hook re-checks coverage
```

### GitHub Integration
```bash
# Create PR - triggers automatic review
gh pr create --title "feat: new feature" --body "Description"

# Workflow automatically:
# 1. Analyzes code quality
# 2. Applies fixes if possible
# 3. Posts review with detailed feedback
# 4. Adds appropriate labels
# 5. Approves or requests changes
```

### Manual Workflow Trigger
```bash
# Run automation on specific PR
gh workflow run pr-auto-review-fix.yml -f pr_number=123
```

## Configuration Files

### `.pre-commit-config.yaml`
- Defines all pre-commit hooks
- Coverage check configuration
- Code formatting rules
- ADR validation

### `.github/workflows/pr-auto-review-fix.yml`
- GitHub Actions workflow definition
- Auto-fix logic
- PR review automation
- Quality metrics collection

## Monitoring and Metrics

The system tracks:
- **Coverage trends**: Before/after automation
- **Auto-fix effectiveness**: Issues resolved automatically
- **PR quality**: Time to approval, review cycles
- **Developer productivity**: Reduced manual review overhead

## Benefits

1. **Quality Assurance**
   - Prevents low-quality code from entering main branch
   - Consistent code formatting and standards
   - Comprehensive test coverage

2. **Developer Experience**
   - Immediate feedback on quality issues
   - Automatic fixing of common problems
   - Clear guidance for manual fixes

3. **Team Productivity**
   - Reduced manual code review overhead
   - Faster PR approval cycle
   - Focus on architectural/business logic reviews

4. **Maintainability**
   - High test coverage ensures safe refactoring
   - Consistent codebase style
   - Automated documentation of decisions (ADRs)

## Future Enhancements

Potential improvements:
- **Security scanning** integration
- **Performance regression** detection
- **Dependency vulnerability** checks
- **Documentation coverage** tracking
- **Custom quality metrics** for trading algorithms

## Troubleshooting

### Pre-Push Hook Issues
```bash
# Skip pre-push hook (emergency only)
git push --no-verify

# Reinstall hooks
pre-commit uninstall
pre-commit install --hook-type pre-push
```

### GitHub Actions Issues
- Check workflow logs in Actions tab
- Ensure required permissions are granted
- Verify secrets and tokens are configured

### Coverage Issues
```bash
# Generate detailed coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html  # View in browser

# Find specific missing lines
pytest --cov=src --cov-report=term-missing
```

---

This automation system ensures the VOYAGER-Trader codebase maintains high quality standards while minimizing manual overhead for developers and reviewers.
