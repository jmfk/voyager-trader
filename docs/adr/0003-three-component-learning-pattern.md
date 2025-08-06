# ADR-0003: Three-Component Autonomous Learning Pattern

## Status

Accepted

## Context

Within the VOYAGER-Trader core architecture (ADR-0002), we need to define how the three key components work together to enable autonomous learning. The original VOYAGER paper demonstrates that the combination of automatic curriculum, skill library, and iterative prompting creates emergent learning capabilities that exceed any single component alone.

### Problem Statement

- Need to define clear interfaces and interaction patterns between components
- Require data flow design that enables continuous learning loops
- Must establish how components share state and coordinate learning
- Need to handle failure cases and recovery in autonomous operation
- Require performance feedback mechanisms between components

### Key Requirements

- Define clear contracts between Curriculum, Skills, and Prompting components
- Enable autonomous learning loops with minimal external intervention
- Support skill discovery, evaluation, and composition patterns
- Provide curriculum adaptation based on skill acquisition progress
- Enable iterative strategy refinement through feedback loops
- Maintain system stability during continuous learning

### Constraints

- Components must remain loosely coupled for testability
- Data flow should be efficient for real-time trading requirements
- State synchronization must be reliable and consistent
- Error handling must prevent cascade failures
- Memory usage must be bounded for long-running operation

## Decision

We will implement a coordinated learning pattern where the three components form autonomous feedback loops through the central VoyagerTrader orchestrator.

### Chosen Approach

**Learning Loop Architecture**:

1. **Curriculum → Prompting Flow**
   - Curriculum generates tasks based on current skill level and market conditions
   - Tasks include context, difficulty level, and success criteria
   - Prompting system receives tasks and generates strategy code
   - Performance feedback flows back to curriculum for next task generation

2. **Prompting → Skills Flow**
   - Successful strategies from prompting are evaluated and validated
   - Validated strategies become skills stored in the library
   - Skills include performance metrics, usage patterns, and composition rules
   - Failed attempts provide negative feedback for prompting improvement

3. **Skills → Curriculum Flow**
   - Available skills inform curriculum about current capabilities
   - Skill performance metrics guide difficulty progression
   - Skill dependencies enable prerequisite-based task ordering
   - Skill gaps identify areas for focused learning

**Data Structures for Coordination**:
- `TradingTask`: Curriculum output with context and criteria
- `TradingSkill`: Skill library entries with performance data
- `PromptContext`: Prompting input with available skills and feedback
- `PerformanceMetrics`: Shared metrics for cross-component feedback

### Alternative Approaches Considered

1. **Direct Component-to-Component Communication**
   - Description: Components directly call each other's methods
   - Pros: Simple, direct communication, lower latency
   - Cons: Tight coupling, hard to test, circular dependencies
   - Why rejected: Violates loose coupling principles

2. **Event-Driven Architecture**
   - Description: Components communicate through events and message queues
   - Pros: Loose coupling, good for distributed systems
   - Cons: Complexity, harder to debug, potential message loss
   - Why rejected: Overkill for single-process learning system

3. **Shared State Repository**
   - Description: All components read/write to central state store
   - Pros: Simple state management, easy to add components
   - Cons: Race conditions, state corruption risks, unclear ownership
   - Why rejected: Makes debugging difficult and violates encapsulation

## Consequences

### Positive Consequences

- Clear data flow enables predictable learning progression
- Loose coupling allows independent component evolution
- Feedback loops enable emergent learning capabilities
- Centralized orchestration provides system-wide coordination
- Performance metrics flow enables adaptive learning
- Error isolation prevents cascade failures

### Negative Consequences

- Orchestrator becomes central point of complexity
- More complex initial implementation than direct communication
- Requires careful state synchronization
- Debugging distributed learning behavior can be challenging

### Neutral Consequences

- Component interfaces need comprehensive documentation
- Testing requires both unit and integration approaches
- Performance monitoring needs cross-component visibility
- Learning behavior emerges from component interactions

## Implementation

### Implementation Steps

1. ✅ Define shared data structures (TradingTask, TradingSkill, PromptContext)
2. ✅ Implement component initialization and lifecycle management
3. ✅ Create learning loop coordination in VoyagerTrader
4. ✅ Add performance metrics collection and sharing
5. ✅ Implement error handling and recovery mechanisms
6. 🔄 Add integration tests for learning loop behavior
7. ⏳ Performance optimization for continuous operation
8. ⏳ Monitoring and observability for learning progression

### Success Criteria

- Components can operate together in continuous learning loops
- Skills are successfully acquired and reused across tasks
- Curriculum adapts difficulty based on skill acquisition
- System maintains stable operation during autonomous learning
- Performance metrics show improvement over time
- Error conditions are handled gracefully without system failure

### Timeline

- Core pattern implementation: Complete (January 2025)
- Integration testing: Within 1 week
- Performance optimization: Within 2 weeks
- Production validation: Within 1 month

## Related

- ADR-0002: VOYAGER-Trader Core Architecture
- Original VOYAGER Paper: Three-component autonomous learning
- Implementation: `src/voyager_trader/core.py` (orchestration logic)
- Components: `src/voyager_trader/{curriculum,skills,prompting}.py`

## Notes

This pattern is the heart of the autonomous learning capability. The success of the entire system depends on how well these components coordinate to create emergent learning behavior.

The implementation follows the proven VOYAGER pattern but adapts it specifically for the trading domain. Key adaptations include:
- Trading-specific task generation in curriculum
- Financial strategy representation in skills
- Market-aware prompting contexts

Future enhancements should focus on:
- More sophisticated skill composition patterns
- Advanced curriculum difficulty algorithms
- Enhanced prompting strategies for financial domain
- Performance optimization for high-frequency operation

The learning loop behavior will be the primary indicator of system success and should be continuously monitored and optimized.

---

**Date**: 2025-01-06
**Author(s)**: Claude Code
**Reviewers**: [To be assigned]
