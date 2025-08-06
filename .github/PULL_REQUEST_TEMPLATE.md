# Pull Request

## Summary
<!-- Provide a brief description of the changes in this PR -->

## Type of Change
<!-- Mark the relevant option with an "x" -->

- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation (documentation only changes)
- [ ] 🎨 Style (formatting, missing semi colons, etc; no code change)
- [ ] ♻️ Refactoring (no functional changes, no API changes)
- [ ] ⚡ Performance (code changes that improve performance)
- [ ] ✅ Test (adding missing tests, refactoring tests; no production code change)
- [ ] 🔧 Chore (updating build tasks, package manager configs, etc; no production code change)

## Related Issues
<!-- Link to related issues using "Fixes #123" or "Closes #123" -->

## Architecture Changes
<!-- If this PR affects system architecture, answer the following: -->

- [ ] **ADR Required**: This change affects system architecture and requires an ADR
- [ ] **ADR Updated**: I have updated or created relevant ADR(s) in `docs/adr/`
- [ ] **No ADR Needed**: This change does not affect system architecture

### SOLID Principles Compliance
<!-- Check all that apply -->

- [ ] **Single Responsibility**: Classes/modules have a single, well-defined purpose
- [ ] **Open/Closed**: Code is open for extension but closed for modification
- [ ] **Liskov Substitution**: Subtypes are substitutable for their base types
- [ ] **Interface Segregation**: Clients are not forced to depend on unused interfaces
- [ ] **Dependency Inversion**: High-level modules don't depend on low-level modules

## Testing
<!-- Describe the testing you've performed -->

- [ ] **Unit Tests**: Added/updated unit tests for new functionality
- [ ] **Integration Tests**: Added/updated integration tests if needed
- [ ] **Manual Testing**: Manually tested the changes
- [ ] **Coverage**: Test coverage meets minimum requirements (80%+)

### Test Results
```
# Paste test results here
pytest tests/ -v --cov=src
```

## Code Quality Checklist
<!-- Verify code quality requirements -->

- [ ] **Linting**: Code passes flake8 linting (`flake8 src/ tests/`)
- [ ] **Formatting**: Code is formatted with Black (`black src/ tests/`)
- [ ] **Type Hints**: Added type hints where appropriate (`mypy src/`)
- [ ] **Documentation**: Updated docstrings and comments as needed
- [ ] **Conventional Commits**: Commit messages follow conventional format

## Security Considerations
<!-- Address any security implications -->

- [ ] **No Secrets**: No secrets, API keys, or sensitive data in the code
- [ ] **Security Scan**: Code passes security scanning (bandit, safety)
- [ ] **Dependencies**: No new dependencies with known vulnerabilities
- [ ] **Input Validation**: Proper input validation and sanitization

## Deployment Notes
<!-- Any special deployment considerations -->

- [ ] **Database Changes**: No database schema changes
- [ ] **Environment Variables**: No new environment variables required
- [ ] **Breaking Changes**: No breaking changes to existing APIs
- [ ] **Migration Required**: No data migration required

## Reviewer Notes
<!-- Additional context for reviewers -->

### Areas of Focus
<!-- Highlight specific areas you'd like reviewers to focus on -->

### Questions for Reviewers
<!-- Any specific questions or concerns you have -->

---

## Checklist for Reviewers

### Code Review
- [ ] Code follows project conventions and style guidelines
- [ ] Logic is correct and handles edge cases appropriately
- [ ] Error handling is appropriate and informative
- [ ] Performance considerations have been addressed
- [ ] Security best practices are followed

### Architecture Review
- [ ] Changes align with existing architecture patterns
- [ ] SOLID principles are maintained
- [ ] DRY, KISS, and YAGNI principles are followed
- [ ] ADRs are updated if architectural changes are made

### Testing Review
- [ ] Tests are comprehensive and cover the new functionality
- [ ] Tests are well-structured and maintainable
- [ ] Test coverage meets project standards
- [ ] Integration tests are included where appropriate

### Documentation Review
- [ ] Code is well-documented with clear docstrings
- [ ] README or other documentation is updated as needed
- [ ] ADRs are properly formatted and complete
