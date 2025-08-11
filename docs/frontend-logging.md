# Frontend Logging System

## Overview

The VoyagerTrader Admin Interface uses a centralized logging system that provides structured, consistent logging across all React components. This system replaces ad-hoc `console.error` calls with proper logging that includes context, levels, and can be configured for different environments.

## Logger Features

- **Multiple Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Structured Logging**: Consistent format with timestamps and context
- **Component-Specific Loggers**: Easy identification of log sources
- **Environment Configuration**: Adjustable log levels via environment variables
- **External Service Integration**: Ready for production logging services
- **Contextual Information**: Rich metadata for debugging

## Usage

### Basic Logging

```javascript
import { createLogger } from '../utils/logger';

const MyComponent = () => {
  const logger = createLogger('MyComponent');

  const handleAction = async () => {
    try {
      // Your code here
      logger.info('Action completed successfully');
    } catch (error) {
      logger.error('Action failed', { error: error.message });
    }
  };
};
```

### Specialized Logging Methods

```javascript
// API Error Logging
logger.apiError('/api/endpoint', error, response);

// Component Error Logging  
logger.componentError('MyComponent', error, props);

// User Action Tracking
logger.userAction('button-click', { buttonId: 'submit' });
```

### Log Levels

- **DEBUG**: Detailed information for development and debugging
- **INFO**: General information about application flow
- **WARNING**: Something unexpected happened but application continues
- **ERROR**: An error occurred that should be investigated
- **CRITICAL**: A serious error occurred that may require immediate attention

## Configuration

### Environment Variables

Set in `.env` file or environment:

```bash
# Control log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
REACT_APP_LOG_LEVEL=INFO
```

### Development vs Production

**Development:**
```bash
REACT_APP_LOG_LEVEL=DEBUG
```
Shows all logs for debugging.

**Production:**
```bash
REACT_APP_LOG_LEVEL=WARNING
```
Shows only warnings, errors, and critical issues to reduce noise.

## Log Format

Logs follow a consistent format:

```
[2024-01-15T10:30:45.123Z] ERROR VoyagerTrader-Admin:Dashboard: API call failed: /api/status [{"endpoint":"/api/status","error":"Network Error"}]
```

Format breakdown:
- `[timestamp]` - ISO 8601 timestamp
- `LEVEL` - Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `VoyagerTrader-Admin:Component` - Logger name with component
- `message` - Human-readable message
- `[context]` - JSON context data (optional)

## Component Integration

### Existing Components

All major components have been updated to use proper logging:

- **Dashboard.js**: System status loading, control actions, user interactions
- **LogsPanel.js**: System log loading and display
- **SkillsTable.js**: Skills data loading with metrics
- **PerformanceChart.js**: Performance data loading with statistics

### Adding Logging to New Components

1. Import the logger:
```javascript
import { createLogger } from '../utils/logger';
```

2. Create component-specific logger:
```javascript
const logger = createLogger('YourComponent');
```

3. Replace console calls with proper logging:
```javascript
// Instead of:
console.error('Something failed:', error);

// Use:
logger.error('Something failed', { error: error.message });
```

## External Service Integration

The logging system is ready for production integration with external services:

### Supported Patterns

```javascript
// Backend API logging endpoint
fetch('/api/logs/frontend', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    level: 'ERROR',
    message: 'Frontend error occurred',
    context: { component: 'Dashboard', error: 'API timeout' },
    timestamp: new Date().toISOString(),
    userAgent: navigator.userAgent,
    url: window.location.href
  })
});
```

### Integration Options

- **LogRocket**: Frontend session replay and logging
- **Sentry**: Error tracking and performance monitoring
- **DataDog**: Application performance monitoring
- **Custom Backend**: Send logs to your own API endpoint

## Best Practices

### 1. Use Appropriate Log Levels

```javascript
logger.debug('Detailed debugging info');     // Development only
logger.info('Normal application flow');      // General information
logger.warning('Unexpected but recoverable'); // Needs attention
logger.error('Error occurred');              // Investigate
logger.critical('System in bad state');      // Immediate action
```

### 2. Provide Context

```javascript
// Good - includes helpful context
logger.error('Failed to load user data', {
  userId: user.id,
  endpoint: '/api/users',
  statusCode: response.status
});

// Poor - minimal information
logger.error('Failed to load data');
```

### 3. Log User Actions

```javascript
logger.userAction('system-start', { 
  timestamp: new Date().toISOString(),
  userRole: 'admin' 
});
```

### 4. Use Specialized Methods

```javascript
// For API errors
logger.apiError('/api/status', error);

// For component errors  
logger.componentError('Dashboard', error, { props });
```

## Troubleshooting

### Logs Not Appearing

1. Check log level configuration:
```javascript
// In browser console:
console.log('Current log level:', process.env.REACT_APP_LOG_LEVEL);
```

2. Verify logger import:
```javascript
import { createLogger } from '../utils/logger';
```

3. Check browser developer tools console

### Too Many/Few Logs

Adjust the log level:

```bash
# Show more logs
REACT_APP_LOG_LEVEL=DEBUG

# Show fewer logs  
REACT_APP_LOG_LEVEL=ERROR
```

### Production Integration Issues

1. Verify external service configuration
2. Check network connectivity for log submission
3. Review CORS settings if sending logs to different domain
4. Monitor for log submission failures (silently handled to avoid loops)

## Migration from console.error

All `console.error` calls have been replaced with proper logging:

### Before
```javascript
console.error('Failed to load system status:', error);
```

### After
```javascript
logger.apiError('/api/status', error);
// or
logger.error('Failed to load system status', { error: error.message });
```

This provides better structure, context, and production-ready logging capabilities.