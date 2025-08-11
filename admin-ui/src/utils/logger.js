/**
 * Centralized logging service for VoyagerTrader Admin Interface.
 * 
 * Provides structured logging with different levels and consistent formatting.
 * In production, can be configured to send logs to external services.
 */

// Log levels (aligned with backend Python logging)
const LogLevel = {
  DEBUG: 0,
  INFO: 1,
  WARNING: 2,
  ERROR: 3,
  CRITICAL: 4
};

class Logger {
  constructor(name = 'VoyagerTrader-Admin') {
    this.name = name;
    this.level = this.getLogLevel();
  }

  getLogLevel() {
    // Check environment variable or use default
    const envLevel = process.env.REACT_APP_LOG_LEVEL || 'INFO';
    return LogLevel[envLevel.toUpperCase()] || LogLevel.INFO;
  }

  formatMessage(level, message, context = null) {
    const timestamp = new Date().toISOString();
    const contextStr = context ? ` [${JSON.stringify(context)}]` : '';
    return `[${timestamp}] ${level} ${this.name}: ${message}${contextStr}`;
  }

  log(level, levelName, message, context = null) {
    if (level < this.level) return;

    const formattedMessage = this.formatMessage(levelName, message, context);
    
    // Use appropriate console method based on level
    switch (level) {
      case LogLevel.DEBUG:
        console.debug(formattedMessage);
        break;
      case LogLevel.INFO:
        console.info(formattedMessage);
        break;
      case LogLevel.WARNING:
        console.warn(formattedMessage);
        break;
      case LogLevel.ERROR:
      case LogLevel.CRITICAL:
        console.error(formattedMessage);
        break;
      default:
        console.log(formattedMessage);
    }

    // In production, could send to external logging service here
    this.sendToExternalService(level, levelName, message, context);
  }

  sendToExternalService(level, levelName, message, context) {
    // Placeholder for external logging service integration
    // Could send to services like:
    // - Backend API logging endpoint
    // - LogRocket, Sentry, DataDog, etc.
    
    if (process.env.NODE_ENV === 'production' && level >= LogLevel.ERROR) {
      // Example: Send critical errors to backend
      // fetch('/api/logs/frontend', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({
      //     level: levelName,
      //     message,
      //     context,
      //     timestamp: new Date().toISOString(),
      //     userAgent: navigator.userAgent,
      //     url: window.location.href
      //   })
      // }).catch(() => {}); // Silently fail to avoid logging loops
    }
  }

  debug(message, context = null) {
    this.log(LogLevel.DEBUG, 'DEBUG', message, context);
  }

  info(message, context = null) {
    this.log(LogLevel.INFO, 'INFO', message, context);
  }

  warning(message, context = null) {
    this.log(LogLevel.WARNING, 'WARNING', message, context);
  }

  error(message, context = null) {
    this.log(LogLevel.ERROR, 'ERROR', message, context);
  }

  critical(message, context = null) {
    this.log(LogLevel.CRITICAL, 'CRITICAL', message, context);
  }

  // Convenience methods for specific use cases
  apiError(endpoint, error, response = null) {
    const context = {
      endpoint,
      error: error.message || error.toString(),
      response: response ? {
        status: response.status,
        statusText: response.statusText,
        data: response.data
      } : null
    };
    this.error(`API call failed: ${endpoint}`, context);
  }

  componentError(componentName, error, props = null) {
    const context = {
      component: componentName,
      error: error.message || error.toString(),
      props: props || {}
    };
    this.error(`Component error: ${componentName}`, context);
  }

  userAction(action, details = null) {
    const context = {
      action,
      timestamp: new Date().toISOString(),
      url: window.location.href,
      ...details
    };
    this.info(`User action: ${action}`, context);
  }
}

// Create default logger instance
const defaultLogger = new Logger('VoyagerTrader-Admin');

// Create component-specific loggers
export const createLogger = (name) => new Logger(`VoyagerTrader-Admin:${name}`);

// Export both default logger and factory function
export default defaultLogger;
export { Logger, LogLevel };

// Convenience exports for common patterns
export const apiLogger = createLogger('API');
export const componentLogger = createLogger('Components');
export const authLogger = createLogger('Auth');