# ADR-0004: Dependency Injection Framework

## Status

Proposed

## Context

The VOYAGER-Trader system requires a robust dependency management strategy to support autonomous operation, testing, and configuration flexibility. As the system grows in complexity with multiple components (curriculum, skills, prompting) and external integrations (market data, brokers, LLM providers), we need a consistent approach for managing dependencies and enabling testability.

### Problem Statement

- Need to manage complex dependencies between core components and external services
- Require testability through dependency substitution and mocking
- Must support different configurations for development, testing, and production environments
- Need to handle optional dependencies and feature flags for different trading environments
- Require clean initialization order for autonomous system startup

### Key Requirements

- Enable clean separation between business logic and infrastructure concerns
- Support configuration-driven dependency resolution for different environments
- Provide testability through dependency injection and interface contracts
- Support optional dependencies for different market data providers and brokers
- Maintain simplicity for the autonomous trading domain
- Enable lazy loading for expensive resources (LLM connections, market data feeds)

### Constraints

- Python ecosystem limitations compared to Java/C# DI frameworks
- Performance requirements for real-time trading operations
- Memory efficiency for long-running autonomous operation
- Simplicity requirements to avoid over-engineering
- Must integrate with existing VOYAGER three-component architecture

## Decision

We will implement a lightweight dependency injection approach using Python's built-in capabilities, avoiding heavy external frameworks while providing sufficient flexibility for testing and configuration management.

### Chosen Approach

**Manual Dependency Injection with Factory Pattern**:

1. **Service Interfaces**: Define abstract base classes for external dependencies
   - `MarketDataProvider` interface for market data sources
   - `BrokerInterface` for trading execution
   - `LLMProvider` interface for AI model interactions
   - `PersistenceLayer` interface for data storage

2. **Factory Classes**: Create factory classes for complex dependency graphs
   - `ServiceFactory` - main factory for creating configured instances
   - Environment-specific factories (DevFactory, ProdFactory, TestFactory)

3. **Configuration-Driven**: Use `TradingConfig` to drive dependency selection
   - Configuration specifies which implementations to use
   - Support for feature flags and optional components

4. **Component Initialization**: Modify core components to accept injected dependencies
   - Constructor injection for required dependencies
   - Property injection for optional dependencies
   - Maintain backward compatibility with current initialization

### Alternative Approaches Considered

1. **dependency-injector Framework**
   - Description: Use established Python DI framework like `dependency-injector`
   - Pros: Full-featured, proven, excellent tooling
   - Cons: Heavy dependency, learning curve, potential over-engineering
   - Why rejected: Adds complexity without significant benefit for our use case

2. **Pure Constructor Injection**
   - Description: Only use constructor parameters, no framework
   - Pros: Simple, explicit, easy to understand
   - Cons: Becomes unwieldy with many dependencies, hard to configure
   - Why rejected: Too rigid for configuration flexibility needs

3. **Service Locator Pattern**
   - Description: Central registry where components look up dependencies
   - Pros: Simple implementation, flexible at runtime
   - Cons: Hidden dependencies, harder to test, violates explicit dependencies principle
   - Why rejected: Makes testing difficult and dependencies less transparent

## Consequences

### Positive Consequences

- Clean testability through dependency substitution
- Configuration flexibility for different environments
- Explicit dependency contracts through interfaces
- Supports autonomous system requirements (lazy loading, optional components)
- Maintains Python idioms and simplicity
- Easy to understand and debug
- No external dependencies for DI framework

### Negative Consequences

- Manual wiring requires more initial setup code
- Less sophisticated than full DI frameworks
- Potential for configuration errors without compile-time checking
- More code to maintain for factory classes

### Neutral Consequences

- Dependency management becomes more explicit in codebase
- Configuration files become more important for system behavior
- Testing setup requires factory configuration
- Component initialization order becomes more critical

## Implementation

### Implementation Steps

1. Define service interfaces for external dependencies
2. Create base `ServiceFactory` class with configuration-driven instantiation
3. Implement environment-specific factory subclasses
4. Modify core components to accept injected dependencies
5. Update `TradingConfig` to include dependency selection options
6. Create test factories for unit testing
7. Update initialization in `VoyagerTrader` to use factory pattern
8. Add examples and documentation for adding new dependencies

### Success Criteria

- All external dependencies can be substituted for testing
- Different configurations can be loaded without code changes
- Component initialization succeeds in all environments (dev, test, prod)
- Performance overhead is negligible for real-time trading
- New dependencies can be added without modifying existing components
- Mock implementations work correctly in test suite

### Timeline

- Interface definition: 3 days
- Factory implementation: 5 days
- Component modification: 4 days
- Testing and documentation: 3 days
- Total: ~2 weeks

## Related

- ADR-0002: VOYAGER-Trader Core Architecture
- ADR-0003: Three-Component Autonomous Learning Pattern
- GitHub Issue #2: ADR: Core System Architecture Design
- Implementation: `src/voyager_trader/dependencies.py` (new)
- Configuration: `src/voyager_trader/core.py` (TradingConfig updates)

## Notes

This approach balances simplicity with the flexibility needed for the autonomous trading domain. The manual DI approach is well-suited for Python and avoids the complexity overhead of full DI frameworks while still providing the key benefits of testability and configuration flexibility.

The factory pattern aligns well with the existing VOYAGER architecture and allows each component to receive its dependencies without tight coupling. This will be especially important as we add market data providers, broker integrations, and different LLM providers.

Future considerations:
- May need to add lifecycle management for long-lived connections
- Could evolve to include dependency health checking for autonomous operation
- May need retry/fallback patterns for critical dependencies

---

**Date**: 2025-08-06
**Author(s)**: Claude Code
**Reviewers**: [To be assigned]
