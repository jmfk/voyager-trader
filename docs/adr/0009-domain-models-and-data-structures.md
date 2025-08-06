# ADR-0009: Domain Models and Data Structures Design

## Status

Proposed

## Context

The VOYAGER-Trader system requires comprehensive domain models that represent the trading system's fundamental concepts following Domain-Driven Design principles. These models must support autonomous trading operations, continuous learning through VOYAGER's three components, and efficient data persistence as defined in ADR-0008.

### Problem Statement

How do we design domain models that accurately represent trading concepts, support VOYAGER's learning capabilities, integrate with our polyglot persistence architecture, and provide type safety and validation for autonomous operations?

### Key Requirements

- Market data models supporting real-time and historical data (OHLCV, tick data, order book)
- Trading entities for portfolio management (Portfolio, Position, Order, Trade)
- Strategy models enabling composable trading logic (Signal, Rule, Strategy)  
- Learning entities supporting VOYAGER's skill acquisition (Skill, Experience, Knowledge)
- VOYAGER-specific models for autonomous operation (Curriculum, Environment, Agent)
- Immutable data structures for thread-safe autonomous operation
- Comprehensive validation using Pydantic for data integrity
- Serialization support for persistence and API communication
- Type safety with full type hints for reliable autonomous operation

### Constraints

- Must align with polyglot persistence architecture (SQLite, InfluxDB, file storage)
- Python 3.10+ type system compatibility
- Performance requirements for real-time trading operations
- Memory efficiency for large market datasets
- Thread safety for concurrent autonomous operations
- Pydantic validation for data integrity and API compatibility
- Immutable designs where appropriate for financial data integrity

## Decision

We will implement a comprehensive domain model hierarchy using Pydantic v2 for validation and serialization, following Domain-Driven Design principles with clear separation between entities, value objects, and aggregates, optimized for the polyglot persistence architecture.

### Chosen Approach

**Domain Model Architecture**:

1. **Market Data Models** (Time-series optimized for InfluxDB):
   - `OHLCV`: Open, High, Low, Close, Volume data with timestamps
   - `TickData`: Individual trade ticks with microsecond precision
   - `OrderBook`: Bid/ask levels with depth and timestamp
   - `MarketEvent`: General market events and news integration

2. **Trading Entity Models** (Relational optimized for SQLite):
   - `Portfolio`: Aggregate root managing positions and cash
   - `Position`: Individual holdings with entry/exit tracking
   - `Order`: Trading orders with lifecycle management
   - `Trade`: Executed trades with performance metrics
   - `Account`: Trading account with risk parameters

3. **Strategy Models** (Relational with file artifacts):
   - `Strategy`: Composable trading strategies with metadata
   - `Signal`: Trading signals with confidence and timing
   - `Rule`: Individual trading rules for strategy composition
   - `Backtest`: Strategy performance results and metrics

4. **Learning Models** (VOYAGER-specific, relational):
   - `Skill`: Learned trading skills with code and metadata
   - `Experience`: Trading experiences with outcomes
   - `Knowledge`: Accumulated trading knowledge and patterns
   - `Performance`: Detailed performance tracking and analysis

5. **VOYAGER System Models** (Relational with configuration):
   - `Curriculum`: Learning progression and task management
   - `Environment`: Trading environment state and context
   - `Agent`: Autonomous agent state and capabilities
   - `Task`: Individual learning tasks with objectives

6. **Common Infrastructure**:
   - `BaseModel`: Pydantic base with common functionality
   - `Entity`: Base class for domain entities with identity
   - `ValueObject`: Base class for immutable value objects
   - `AggregateRoot`: Base class for aggregate boundaries

### Alternative Approaches Considered

1. **Dataclasses with Custom Validation**
   - Description: Use Python dataclasses with custom validation logic
   - Pros: Simpler, fewer dependencies, more control over validation
   - Cons: More code to maintain, no automatic serialization, poor API integration
   - Why rejected: Pydantic provides superior validation and serialization features

2. **SQLAlchemy ORM Models as Domain Models**
   - Description: Use SQLAlchemy models directly as domain entities
   - Pros: Single model definition, automatic persistence mapping
   - Cons: Tight coupling to persistence, poor separation of concerns, database-specific constraints
   - Why rejected: Violates clean architecture principles, reduces testability

3. **Protocol-Based Models (Structural Typing)**
   - Description: Use Python protocols for duck typing instead of concrete classes
   - Pros: Flexible interfaces, easier testing, loose coupling
   - Cons: No runtime validation, poor serialization support, harder debugging
   - Why rejected: Autonomous system needs runtime validation for safety

## Consequences

The Pydantic-based domain model architecture will provide strong validation and serialization capabilities while maintaining clean separation from persistence concerns.

### Positive Consequences

- Strong runtime validation prevents data corruption in autonomous operations
- Automatic JSON serialization supports API development and caching
- Type safety improves reliability for autonomous trading operations
- Clear domain model hierarchy follows DDD principles
- Immutable value objects ensure thread safety for concurrent operations
- Easy integration with FastAPI for future API development
- Rich metadata support for learning and performance tracking

### Negative Consequences

- Additional dependency on Pydantic increases complexity
- Runtime validation overhead may impact high-frequency operations
- Memory usage higher than plain dataclasses
- Learning curve for team members unfamiliar with Pydantic

### Neutral Consequences

- Model definitions are more verbose than simple dataclasses
- Serialization format is JSON-based (may need binary formats for performance)
- Validation rules need maintenance as requirements evolve

## Implementation

### Implementation Steps

1. Create base model infrastructure with Pydantic BaseModel extensions
2. Implement market data models with time-series optimizations
3. Implement trading entity models with business rule validation  
4. Create strategy models supporting composition and backtesting
5. Implement learning models for VOYAGER skill management
6. Create VOYAGER system models for autonomous operation
7. Add comprehensive validation rules and error handling
8. Implement serialization adapters for persistence layer
9. Create model factories for testing and development

### Success Criteria

- All domain concepts are clearly modeled with appropriate validation
- Models support efficient serialization to/from persistence formats
- Type checking passes without errors using mypy
- Models are immutable where required for thread safety
- Validation prevents invalid data states in autonomous operations
- Performance benchmarks meet real-time trading requirements
- 100% test coverage for all model validation and serialization

### Timeline

- Base infrastructure: 1 day
- Market data models: 1 day
- Trading entity models: 2 days
- Strategy models: 2 days  
- Learning models: 2 days
- VOYAGER system models: 1 day
- Validation and serialization: 1 day
- Testing and documentation: 2 days
- Total: ~2 weeks

## Related

- ADR-0002: VOYAGER-Trader Core Architecture
- ADR-0008: Data Persistence Layer Design
- GitHub Issue #3: Core: Data Models and Domain Entities
- Implementation: `src/voyager_trader/models/` (new directory)
- Tests: `tests/models/` (new directory)

## Notes

**Pydantic v2 Benefits for Trading Systems**:
- Field validation ensures data integrity for financial calculations
- Automatic type coercion handles API data conversion safely
- JSON Schema generation supports API documentation
- Performance improvements in v2 are suitable for trading applications
- Built-in serialization handles complex nested models efficiently

**Model Organization Strategy**:
```
src/voyager_trader/models/
├── __init__.py           # Public API exports
├── base.py               # Base classes and infrastructure
├── market.py             # Market data models
├── trading.py            # Trading entities
├── strategy.py           # Strategy and signal models  
├── learning.py           # VOYAGER learning models
├── system.py             # VOYAGER system models
└── types.py              # Common value objects and enums
```

**Validation Strategy**:
- Field-level validation for data types and ranges
- Model-level validation for business rules and constraints
- Custom validators for financial calculations and risk checks
- Enum validation for controlled vocabularies
- Cross-field validation for complex business rules

**Performance Considerations**:
- Use `__slots__` for memory efficiency in high-frequency models
- Lazy loading for expensive computed properties
- Caching for frequently accessed calculated fields
- Batch validation for bulk data processing
- Optional validation bypass for internal operations

The domain models will serve as the foundation for all business logic while maintaining clean separation from infrastructure concerns through the repository pattern defined in ADR-0008.

---

**Date**: 2025-08-06
**Author(s)**: Claude Code
**Reviewers**: [To be assigned]
