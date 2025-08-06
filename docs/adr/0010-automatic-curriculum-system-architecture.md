# ADR-0010: Automatic Curriculum System Architecture

## Status

Accepted

## Context

VOYAGER-Trader requires an automatic curriculum system that progressively generates learning objectives for the trading agent, inspired by VOYAGER's curriculum learning approach. The system must adaptively create trading challenges, track performance, and adjust difficulty based on agent capabilities and market conditions.

### Problem Statement

We need to implement the automatic curriculum component of VOYAGER-Trader that can autonomously generate progressive learning tasks for the trading agent, similar to how VOYAGER generates increasingly complex Minecraft objectives.

### Key Requirements

- Progressive difficulty scaling from basic to advanced trading strategies
- Adaptive curriculum based on agent performance and learning progress
- Market condition awareness in task generation
- Task dependency management and prerequisite handling
- Performance tracking and success/failure analysis
- Curriculum persistence across system restarts
- Integration with existing skill library and experience tracking
- Support for different curriculum strategies (progressive, adaptive, exploratory, etc.)

### Constraints

- Must integrate with existing domain models (Task, Agent, Curriculum entities)
- Must follow existing architecture patterns (Domain-Driven Design, dependency injection)
- Must support autonomous operation without human intervention
- Must be testable and maintainable
- Must handle various market conditions and trading environments
- Performance must not impact real-time trading operations

## Decision

We will implement a modular automatic curriculum system consisting of five core components that work together to generate, assess, track, and adapt learning objectives for the trading agent.

### Chosen Approach

**Architecture Components:**

1. **CurriculumGenerator**: Creates new trading objectives and tasks
   - Generates tasks based on current difficulty level and agent capabilities
   - Considers market conditions and available trading opportunities
   - Manages task templates and objective patterns
   - Supports different generation strategies (progressive, exploratory, etc.)

2. **DifficultyAssessor**: Evaluates challenge complexity and appropriateness
   - Analyzes task complexity across multiple dimensions
   - Considers market volatility, strategy complexity, and risk levels
   - Provides difficulty scoring for task validation
   - Ensures appropriate progression paths

3. **ProgressTracker**: Monitors learning success rates and performance metrics
   - Tracks task completion rates and success patterns
   - Maintains historical performance data
   - Identifies learning trends and plateau detection
   - Generates performance reports for adaptive decisions

4. **AdaptiveEngine**: Modifies curriculum based on performance feedback
   - Analyzes performance data to make curriculum adjustments
   - Implements adaptation strategies (difficulty scaling, focus areas, etc.)
   - Handles both positive and negative performance trends
   - Maintains learning momentum and prevents stagnation

5. **ContextAnalyzer**: Considers market conditions in curriculum design
   - Analyzes current market regime and conditions
   - Matches appropriate learning objectives to market states
   - Provides market context for task generation
   - Ensures curriculum relevance to current trading environment

**Integration Approach:**

- Components interact through well-defined interfaces
- Central CurriculumService orchestrates component interactions
- Domain events communicate state changes between components
- Persistence layer handles curriculum state and task history
- Configuration system allows tuning of curriculum parameters

### Alternative Approaches Considered

1. **Monolithic Curriculum Engine**
   - Description: Single large class handling all curriculum functionality
   - Pros: Simpler initial implementation, centralized logic
   - Cons: Poor maintainability, difficult testing, tight coupling
   - Why rejected: Violates Single Responsibility Principle, hard to extend

2. **Rule-Based Static Curriculum**
   - Description: Pre-defined curriculum with fixed progression rules
   - Pros: Predictable behavior, easier to understand
   - Cons: Not adaptive to individual agent performance, limited flexibility
   - Why rejected: Doesn't align with VOYAGER's adaptive learning philosophy

3. **Machine Learning-Based Curriculum**
   - Description: Use ML models to generate and adapt curriculum
   - Pros: Potentially more sophisticated adaptation
   - Cons: Complexity, training data requirements, interpretability issues
   - Why rejected: Premature optimization, adds unnecessary complexity for MVP

## Consequences

### Positive Consequences

- Modular design enables independent testing and development of each component
- Clear separation of concerns makes the system maintainable and extensible
- Adaptive nature ensures curriculum stays relevant to agent capabilities
- Market awareness prevents inappropriate tasks during unsuitable conditions
- Performance tracking provides data-driven insights for continuous improvement
- Flexible architecture supports different curriculum strategies and trading styles

### Negative Consequences

- Increased complexity compared to a simple static curriculum
- More components to coordinate and maintain
- Potential performance overhead from adaptive calculations
- Requires comprehensive testing of component interactions
- May need tuning and optimization based on real-world usage patterns

### Neutral Consequences

- Follows established DDD patterns, consistent with existing codebase
- Adds new domain concepts that team must understand and maintain
- Requires documentation and training for proper usage

## Implementation

### Implementation Steps

1. Define curriculum component interfaces and contracts
2. Implement CurriculumGenerator with basic task templates
3. Implement DifficultyAssessor with multi-dimensional scoring
4. Implement ProgressTracker with performance analysis
5. Implement AdaptiveEngine with basic adaptation strategies
6. Implement ContextAnalyzer with market condition detection
7. Create CurriculumService to orchestrate components
8. Add persistence layer for curriculum state
9. Implement comprehensive test suite
10. Add monitoring and observability features

### Success Criteria

- Curriculum generates progressively harder trading tasks
- System adapts to agent's learning progress automatically
- Curriculum persists and resumes correctly across restarts
- Different market conditions trigger appropriate curricula
- Performance metrics guide curriculum evolution effectively
- Comprehensive test coverage (>95%) achieved
- Integration tests validate end-to-end curriculum flow

### Timeline

- Initial implementation: 2-3 weeks
- Testing and refinement: 1 week
- Integration and documentation: 1 week
- Total estimated time: 12-15 hours of development

## Related

- ADR-0003: Three Component Learning Pattern
- ADR-0002: VOYAGER-Trader Core Architecture
- ADR-0009: Domain Models and Data Structures
- GitHub Issue #4: VOYAGER: Automatic Curriculum System
- VOYAGER paper: Research/voyager.md

## Notes

This ADR focuses on the architectural decisions for the curriculum system. Specific implementation details such as task templates, difficulty scoring algorithms, and adaptation strategies will be refined during development based on testing and real-world usage patterns.

The design prioritizes flexibility and maintainability to support future enhancements as we learn more about effective curriculum strategies for autonomous trading agents.

---

**Date**: 2025-08-06
**Author(s)**: Claude Code Assistant
**Reviewers**: TBD
