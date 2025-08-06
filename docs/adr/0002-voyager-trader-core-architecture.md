# ADR-0002: VOYAGER-Trader Core Architecture

## Status

Accepted

## Context

The VOYAGER-Trader project aims to create an autonomous, self-improving trading system inspired by the VOYAGER project's approach to open-ended skill learning. We need to establish a core architectural foundation that supports continuous learning, skill acquisition, and strategy development without human intervention.

### Problem Statement

- Need a system architecture that supports autonomous trading strategy discovery
- Require integration of three key VOYAGER components: automatic curriculum, skill library, and iterative prompting
- Must enable continuous learning and improvement in dynamic market conditions
- System should be modular, testable, and maintainable
- Need clear separation of concerns between different system components

### Key Requirements

- Implement the three core VOYAGER components for trading domain
- Support autonomous operation with minimal human intervention
- Enable strategy composition and reuse through skill library
- Provide progressive learning through automatic curriculum generation
- Support iterative strategy refinement through LLM prompting
- Maintain performance metrics and learning progress tracking
- Ensure modularity and testability of all components

### Constraints

- Must follow SOLID principles for maintainability
- Python-based implementation for ecosystem compatibility
- Configuration-driven to support different trading environments
- Memory and compute efficiency for real-time trading
- Integration with existing financial data providers and brokers

## Decision

We will implement a three-component architecture based on the VOYAGER paper, adapted for autonomous trading:

### Chosen Approach

**Core Architecture Pattern**: The system consists of a central `VoyagerTrader` orchestrator that coordinates three specialized components:

1. **Automatic Curriculum** (`curriculum.py`)
   - Generates progressive trading tasks based on market conditions
   - Adapts difficulty based on agent performance
   - Provides structured learning progression

2. **Skill Library** (`skills.py`)
   - Stores and manages learned trading strategies as reusable skills
   - Supports skill composition and dependency tracking
   - Maintains performance metrics for each skill

3. **Iterative Prompting** (`prompting.py`)
   - Manages LLM interactions for strategy generation
   - Implements iterative refinement based on feedback
   - Handles code generation and validation

**Central Orchestrator** (`core.py`)
   - `VoyagerTrader` class coordinates all components
   - `TradingConfig` provides centralized configuration
   - Maintains system state and performance tracking

### Alternative Approaches Considered

1. **Monolithic Architecture**
   - Description: Single large class handling all functionality
   - Pros: Simple initial implementation, fewer interfaces
   - Cons: Poor maintainability, hard to test, violates SRP
   - Why rejected: Does not scale and makes testing difficult

2. **Microservices Architecture**
   - Description: Separate services for each component
   - Pros: High scalability, independent deployment
   - Cons: Network overhead, complexity for single-machine deployment
   - Why rejected: Overkill for current scope, adds unnecessary complexity

3. **Plugin-based Architecture**
   - Description: Core system with pluggable strategy modules
   - Pros: High extensibility, clean interfaces
   - Cons: More complex initial setup, potential performance overhead
   - Why rejected: VOYAGER's three-component pattern is more proven

## Consequences

### Positive Consequences

- Clear separation of concerns following VOYAGER proven pattern
- Each component can be developed, tested, and maintained independently
- Modular design enables easy extension and modification
- Configuration-driven approach supports different trading environments
- Strong foundation for autonomous learning and strategy development
- Follows SOLID principles improving maintainability

### Negative Consequences

- Initial complexity higher than monolithic approach
- Requires careful coordination between components
- More files and interfaces to maintain
- Potential performance overhead from component interactions

### Neutral Consequences

- Component interfaces need to be well-defined and stable
- Testing strategy needs to cover both unit and integration levels
- Documentation must explain component interactions clearly

## Implementation

### Implementation Steps

1. ✅ Create core module structure with four main files
2. ✅ Implement `TradingConfig` dataclass for centralized configuration
3. ✅ Create `VoyagerTrader` orchestrator class
4. ✅ Implement `AutomaticCurriculum` with task generation framework
5. ✅ Create `SkillLibrary` with persistence and search capabilities
6. ✅ Build `IterativePrompting` system with LLM integration framework
7. ✅ Add comprehensive logging throughout all components
8. ✅ Create initial test suite covering all components
9. 🔄 Add integration points for market data and broker APIs
10. ⏳ Implement actual LLM integration in prompting component

### Success Criteria

- All components can be instantiated and initialized successfully
- Configuration system supports different trading environments
- Skill library can persist and retrieve trading strategies
- Curriculum can generate progressive tasks based on performance
- Prompting system provides framework for LLM integration
- System maintains clear performance metrics and state tracking
- Test coverage exceeds 80% for all components

### Timeline

- Initial implementation: Complete (January 2025)
- Integration with market data: Within 2 weeks
- LLM integration: Within 1 month  
- Production deployment: Within 2 months

## Related

- GitHub Issue: [To be linked with PR]
- VOYAGER Paper: [Original research inspiration]
- Core Implementation: `src/voyager_trader/core.py`
- Component Implementations: `src/voyager_trader/{curriculum,skills,prompting}.py`

## Notes

This architecture establishes the foundation for all future development. The three-component pattern from VOYAGER has been successfully adapted to the trading domain while maintaining the core principles of autonomous learning and skill acquisition.

The modular design allows each component to evolve independently while maintaining clear contracts through the central orchestrator. This will be crucial as we add more sophisticated market integration, risk management, and strategy optimization capabilities.

Future ADRs should document specific implementation decisions within each component, integration patterns with external services, and performance optimization approaches.

---

**Date**: 2025-01-06
**Author(s)**: Claude Code  
**Reviewers**: [To be assigned]
