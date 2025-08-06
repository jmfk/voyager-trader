# Architecture Decision Records (ADR)

## Status

Active - This directory contains Architecture Decision Records for the VOYAGER-Trader project. ADRs document important architectural decisions made during the project's development.

## What is an ADR?

An Architecture Decision Record (ADR) is a document that captures a significant architectural decision along with its context and consequences. ADRs help maintain a historical record of why certain architectural choices were made.

## ADR Lifecycle

1. **Proposed** - Initial proposal for discussion
2. **Accepted** - Decision has been made and is active
3. **Superseded** - Decision has been replaced by a newer ADR
4. **Deprecated** - Decision is no longer recommended but may still be in use

## Naming Convention

ADR files should follow this naming pattern:
```
NNNN-title-of-decision.md
```

Where:
- `NNNN` is a 4-digit sequential number (0001, 0002, etc.)
- `title-of-decision` is a short, descriptive title in kebab-case

## When to Create an ADR

Create an ADR when making decisions about:

- Overall system architecture
- Technology stack choices
- Data storage and persistence strategies
- API design patterns
- Security implementations
- Performance optimization approaches
- Third-party integrations
- Development workflow changes
- Testing strategies

## ADR Requirements for Claude Code

### Before Making Architectural Changes:
1. Check existing ADRs: `ls docs/adr/`
2. If relevant ADR exists, follow it or create superseding ADR
3. If no ADR exists for significant architectural decision, create one
4. Link ADR to GitHub issue: `gh issue create --title "ADR: New Decision" --body "Link to ADR"`

### ADR Creation Process:
1. Copy template: `cp docs/adr/0000-template.md docs/adr/NNNN-your-decision.md`
2. Fill in all sections completely
3. Set status to "Proposed"
4. Create PR with ADR
5. After review/approval, change status to "Accepted"
6. Update related documentation

### When Superseding an ADR:
1. Create new ADR with higher number
2. Reference the superseded ADR
3. Include detailed migration path
4. Update old ADR status to "Superseded"
5. Add link to new ADR in old one
6. Archive old ADR (keep in repository)

## Directory Structure

```
docs/adr/
├── README.md                    # This file
├── 0000-template.md            # ADR template
├── 0001-record-architecture-decisions.md  # Meta-ADR about using ADRs
├── 0002-example-decision.md    # Example ADR
└── archived/                   # Superseded ADRs (optional)
    └── 0001-old-decision.md
```

## Migration Path Documentation

When superseding an ADR, always include a "Migration Path" section with:

1. **Timeline** - When the migration should occur
2. **Steps** - Detailed steps to transition from old to new approach
3. **Risks** - Potential issues during migration
4. **Rollback Plan** - How to revert if needed
5. **Success Criteria** - How to know migration is complete

## Integration with Development Workflow

- ADRs are checked in PR reviews via automated workflow
- Major architectural changes require ADR updates
- New team members should read all accepted ADRs
- ADRs should be referenced in code comments for complex implementations

## Current ADRs

| Number | Title | Status | Date |
|--------|-------|--------|------|
| [0001](0001-record-architecture-decisions.md) | Record Architecture Decisions | Accepted | 2025-01-06 |
| [0002](0002-voyager-trader-core-architecture.md) | VOYAGER-Trader Core Architecture | Accepted | 2025-01-06 |
| [0003](0003-three-component-learning-pattern.md) | Three-Component Autonomous Learning Pattern | Accepted | 2025-01-06 |

## Tools and Automation

- GitHub workflow validates ADR format
- PR reviews check for ADR updates when architectural changes detected
- Claude Code automatically suggests ADR creation for significant changes
