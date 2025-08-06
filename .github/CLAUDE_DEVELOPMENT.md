# Claude Code Development Control Document

This document instructs Claude Code on interactive development workflows for the VOYAGER-Trader project.

## GitHub Integration Workflow

### Issue Management
- Always check existing GitHub issues before starting work: `gh issue list --state open`
- Create new issues for features/bugs: `gh issue create --title "Feature: X" --body "Description"`
- Link commits to issues using: `git commit -m "feat: implement X (closes #123)"`
- Update issue status during development: `gh issue comment 123 --body "Status update"`

### Branch Strategy
- Create feature branches from main: `git checkout -b feature/issue-123-feature-name`
- Keep branches focused on single issues
- Use conventional commit format: `type(scope): description`

### Pull Request Process
1. Push feature branch: `git push -u origin feature/issue-123-feature-name`
2. Create PR: `gh pr create --title "feat: implement X" --body "Closes #123"`
3. Request reviews: `gh pr review --approve` (for self-review after automated checks)
4. Merge after approvals: `gh pr merge --squash`

## Required Commands for Claude Code

### Before Starting Work
```bash
# Check current status
gh issue list --state open
git status
git pull origin main

# Check existing ADRs for architectural guidance
ls docs/adr/
cat docs/adr/README.md

# Create issue if needed
gh issue create --title "Feature: X" --body "Detailed description"

# Create branch
git checkout -b feature/issue-X-name
```

### During Development
```bash
# Run tests first (TDD approach)
python -m pytest tests/ -v

# Install dependencies if needed
pip install -r requirements.txt

# Commit frequently with issue reference
git add .
git commit -m "feat: implement X (refs #123)"
```

### Before PR Creation
```bash
# Ensure all tests pass
python -m pytest tests/ --cov=src/

# Check code quality
flake8 src/ tests/
black src/ tests/ --check
mypy src/

# Push and create PR
git push -u origin feature/issue-123-name
gh pr create --title "feat: implement X" --body "Closes #123"
```

## Interactive Development Guidelines

### Test-First Paradigm
1. **Always write tests first** before implementation
2. Run `python -m pytest tests/ -v` to see failing tests
3. Implement minimal code to make tests pass
4. Refactor while keeping tests green
5. Add integration tests for complex features

### Issue-Driven Development
1. Break work into small, focused GitHub issues
2. Each issue should be completable in 1-2 hours
3. Link all commits to issues using conventional format
4. Update issue comments with progress and blockers
5. Close issues only when fully complete and tested

### Code Quality Gates
- All PRs must pass automated checks
- Code coverage must be ≥90%
- No linting errors allowed
- Type checking must pass
- Documentation must be updated for public APIs
- **ADR compliance must be verified for architectural changes**

## Automation Requirements

Claude Code must always:
1. Check if virtualenv is activated before pip commands
2. Run full test suite before creating PRs
3. Update issue status during development
4. Follow conventional commit format
5. Create PRs with descriptive titles and bodies
6. Ensure code passes all quality checks

## Emergency Procedures

If automated checks fail:
1. Fix issues locally first
2. Run tests again: `python -m pytest tests/ -v`
3. Re-push with force if needed: `git push --force-with-lease`
4. Never merge failing PRs

## State-of-the-Art Design Principles

All code must follow:
- **SOLID principles** (Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion)
- **DRY** (Don't Repeat Yourself)
- **KISS** (Keep It Simple, Stupid)
- **YAGNI** (You Aren't Gonna Need It)
- **Test-Driven Development**
- **Clean Architecture patterns**
- **Dependency Injection**
- **Immutable data structures where possible**

## Architecture Decision Records (ADR)

### ADR Requirements for All Architectural Changes

Claude Code **MUST** follow these ADR requirements:

#### Before Making Architectural Decisions:
1. **Check existing ADRs**: `ls docs/adr/` and review relevant decisions
2. **Evaluate if change requires new ADR**: Any decision affecting system architecture, technology choices, or design patterns
3. **Follow existing ADRs**: Implement according to accepted architectural decisions

#### When Creating New ADRs:
```bash
# Find next ADR number
NEXT_NUM=$(printf "%04d" $(($(ls docs/adr/[0-9]*.md | tail -1 | grep -o '[0-9]\{4\}') + 1)))

# Copy template
cp docs/adr/0000-template.md docs/adr/${NEXT_NUM}-your-decision-title.md

# Edit ADR with all required sections:
# - Status (Proposed initially)
# - Context (problem, requirements, constraints)
# - Decision (chosen approach + alternatives considered)
# - Consequences (positive, negative, neutral)
# - Implementation (steps, criteria, timeline)
```

#### ADR Workflow:
1. **Create ADR** with status "Proposed"
2. **Link to GitHub issue** referencing the ADR
3. **Include ADR in PR** with implementation
4. **After PR approval**, change status to "Accepted"
5. **Reference ADR** in code comments for complex implementations

#### When Superseding Existing ADRs:
1. **Create new ADR** referencing the superseded one
2. **Include detailed Migration Path** section with:
   - Timeline for migration
   - Step-by-step migration steps
   - Risk assessment
   - Rollback plan
   - Success criteria
3. **Update old ADR** status to "Superseded"
4. **Add link** to new ADR in superseded ADR
5. **Keep old ADR** in repository for historical context

#### ADR Enforcement:
- GitHub workflow automatically validates ADR format
- PR reviews check for ADR updates when architectural changes detected
- All ADRs must have required sections: Status, Context, Decision, Consequences
- Superseded ADRs must include Migration Path section
- ADR violations block PR merging

### Examples of When ADR is Required:

**Definitely Requires ADR:**
- New framework or library adoption
- Database schema design decisions
- API design patterns
- Authentication/authorization approaches
- Deployment and infrastructure choices
- Major algorithm or data structure selections
- Integration patterns with external services

**May Require ADR:**
- Significant refactoring approaches
- Performance optimization strategies
- Error handling patterns
- Configuration management approaches

**Does Not Require ADR:**
- Bug fixes without architectural impact
- Minor feature additions following existing patterns
- Code formatting or style changes
- Test additions without new testing strategies