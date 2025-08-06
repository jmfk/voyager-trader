# Test-Driven Development Guidelines for Claude Code

## TDD Cycle for Claude Code

Claude Code must follow this strict TDD cycle:

### 1. Red Phase (Write Failing Test)
```bash
# Always start here - write the test first
# File: tests/test_feature.py
def test_new_feature():
    # Arrange
    setup_data = create_test_data()
    
    # Act
    result = feature_function(setup_data)
    
    # Assert
    assert result.status == "success"
    assert result.data is not None

# Run to confirm it fails
python -m pytest tests/test_feature.py::test_new_feature -v
```

### 2. Green Phase (Make Test Pass)
```bash
# Implement minimal code to make test pass
# File: src/feature.py
def feature_function(data):
    return FeatureResult(status="success", data={})

# Run to confirm it passes
python -m pytest tests/test_feature.py::test_new_feature -v
```

### 3. Refactor Phase (Improve Code)
```bash
# Refactor while keeping tests green
# Run full suite to ensure no regression
python -m pytest tests/ -v
```

## Test Structure Standards

### Test File Organization
```
tests/
├── unit/           # Pure unit tests
├── integration/    # Integration tests
├── e2e/           # End-to-end tests
├── fixtures/      # Test data and fixtures
└── conftest.py    # Pytest configuration
```

### Test Naming Convention
```python
# Pattern: test_[unit_under_test]_[scenario]_[expected_outcome]
def test_portfolio_calculator_with_valid_data_returns_correct_balance():
    pass

def test_trade_executor_with_insufficient_funds_raises_error():
    pass

def test_market_analyzer_with_empty_data_returns_no_signals():
    pass
```

### Test Categories and Requirements

#### Unit Tests (80% of test suite)
```python
# Test individual functions/methods in isolation
# Mock external dependencies
# Fast execution (<1ms per test)

import pytest
from unittest.mock import Mock, patch

def test_calculate_position_size_returns_correct_value():
    # Given
    account_balance = 10000
    risk_percent = 0.02
    
    # When
    result = calculate_position_size(account_balance, risk_percent)
    
    # Then
    assert result == 200
```

#### Integration Tests (15% of test suite)
```python
# Test component interactions
# Use test database/external services
# Medium execution time (<100ms per test)

def test_trading_pipeline_end_to_end():
    # Given
    test_market_data = load_test_data("sample_market.json")
    
    # When
    signals = analyze_market(test_market_data)
    trades = execute_trades(signals)
    
    # Then
    assert len(trades) > 0
    assert all(trade.status == "executed" for trade in trades)
```

#### End-to-End Tests (5% of test suite)
```python
# Test complete user workflows
# Use real services in test environment
# Slow execution (<5s per test)

def test_full_trading_cycle():
    # Test complete trading workflow
    pass
```

## Required Test Fixtures

### Standard Test Data
```python
# tests/fixtures/market_data.py
@pytest.fixture
def sample_market_data():
    return {
        "timestamp": "2025-01-01T00:00:00Z",
        "open": 100.0,
        "high": 105.0,
        "low": 95.0,
        "close": 102.0,
        "volume": 1000000
    }

@pytest.fixture
def sample_portfolio():
    return Portfolio(
        cash=10000,
        positions=[Position("AAPL", 100, 150.0)]
    )
```

## Test Quality Requirements

### Coverage Requirements
- Overall coverage: ≥95%
- Critical path coverage: 100%
- New code coverage: 100%

### Performance Requirements
- Unit tests: <1ms each
- Integration tests: <100ms each
- E2E tests: <5s each
- Full suite: <30s total

### Test Commands for Claude Code
```bash
# Run specific test
python -m pytest tests/test_module.py::test_function -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run only fast tests during development
python -m pytest tests/unit/ -v

# Run full suite before PR
python -m pytest tests/ --cov=src --cov-report=term-missing -v
```

## Error Handling Tests

### Always Test Error Cases
```python
def test_invalid_input_raises_value_error():
    with pytest.raises(ValueError, match="Invalid input"):
        process_data(invalid_input)

def test_network_error_handling():
    with patch('requests.get') as mock_get:
        mock_get.side_effect = ConnectionError()
        
        result = fetch_market_data()
        
        assert result.error == "Network unavailable"
```

## Mocking Guidelines

### Mock External Dependencies
```python
# Always mock external services, APIs, databases
@patch('src.market_api.MarketAPI')
def test_data_fetcher_with_mocked_api(mock_api):
    mock_api.return_value.get_data.return_value = {"price": 100}
    
    result = fetch_current_price("AAPL")
    
    assert result == 100
    mock_api.return_value.get_data.assert_called_once_with("AAPL")
```

## Mandatory Pre-Commit Checks

Claude Code must run these before any commit:

```bash
# 1. Run full test suite
python -m pytest tests/ -v

# 2. Check coverage
python -m pytest tests/ --cov=src --cov-fail-under=95

# 3. Lint code
flake8 src/ tests/

# 4. Format code
black src/ tests/

# 5. Type checking
mypy src/

# Only commit if ALL pass
```

## Test-First Development Flow for Claude Code

1. **Understand requirement** from GitHub issue
2. **Write failing test** that describes expected behavior
3. **Run test** to confirm it fails for right reason
4. **Write minimal code** to make test pass
5. **Run test** to confirm it passes
6. **Refactor code** while keeping test green
7. **Run full suite** to ensure no regression
8. **Commit with issue reference**
9. **Repeat** for next requirement

This ensures all code is thoroughly tested and follows TDD principles.