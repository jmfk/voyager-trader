-- Database schema for VOYAGER Trader persistent storage
-- This schema supports trades, positions, accounts, portfolios, and audit logs
-- Designed for SQLite initially but compatible with PostgreSQL

-- Enable foreign key constraints in SQLite
PRAGMA foreign_keys = ON;

-- Accounts table
CREATE TABLE accounts (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1,
    name TEXT NOT NULL,
    account_type TEXT NOT NULL CHECK (account_type IN ('cash', 'margin', 'retirement')),
    base_currency TEXT NOT NULL DEFAULT 'USD',
    cash_balance_amount DECIMAL(20, 8) NOT NULL DEFAULT 0.0,
    cash_balance_currency TEXT NOT NULL DEFAULT 'USD',
    total_equity_amount DECIMAL(20, 8) NOT NULL DEFAULT 0.0,
    total_equity_currency TEXT NOT NULL DEFAULT 'USD',
    buying_power_amount DECIMAL(20, 8) NOT NULL DEFAULT 0.0,
    buying_power_currency TEXT NOT NULL DEFAULT 'USD',
    maintenance_margin_amount DECIMAL(20, 8) NOT NULL DEFAULT 0.0,
    maintenance_margin_currency TEXT NOT NULL DEFAULT 'USD',
    day_trading_buying_power_amount DECIMAL(20, 8) NULL,
    day_trading_buying_power_currency TEXT NULL,
    max_position_size DECIMAL(5, 2) NOT NULL DEFAULT 10.0 CHECK (max_position_size >= 0 AND max_position_size <= 100),
    max_portfolio_risk DECIMAL(5, 2) NOT NULL DEFAULT 2.0 CHECK (max_portfolio_risk >= 0 AND max_portfolio_risk <= 100),
    daily_loss_limit_amount DECIMAL(20, 8) NULL,
    daily_loss_limit_currency TEXT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    risk_parameters TEXT NOT NULL DEFAULT '{}' -- JSON serialized risk parameters
);

-- Portfolios table
CREATE TABLE portfolios (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1,
    name TEXT NOT NULL,
    account_id TEXT NOT NULL,
    base_currency TEXT NOT NULL DEFAULT 'USD',
    cash_balance_amount DECIMAL(20, 8) NOT NULL DEFAULT 0.0,
    cash_balance_currency TEXT NOT NULL DEFAULT 'USD',
    total_value_amount DECIMAL(20, 8) NOT NULL DEFAULT 0.0,
    total_value_currency TEXT NOT NULL DEFAULT 'USD',
    unrealized_pnl_amount DECIMAL(20, 8) NOT NULL DEFAULT 0.0,
    unrealized_pnl_currency TEXT NOT NULL DEFAULT 'USD',
    realized_pnl_amount DECIMAL(20, 8) NOT NULL DEFAULT 0.0,
    realized_pnl_currency TEXT NOT NULL DEFAULT 'USD',
    daily_pnl_amount DECIMAL(20, 8) NOT NULL DEFAULT 0.0,
    daily_pnl_currency TEXT NOT NULL DEFAULT 'USD',
    max_drawdown DECIMAL(10, 4) NOT NULL DEFAULT 0.0,
    risk_metrics TEXT NOT NULL DEFAULT '{}', -- JSON serialized risk metrics
    performance_metrics TEXT NOT NULL DEFAULT '{}', -- JSON serialized performance metrics
    FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE CASCADE
);

-- Orders table
CREATE TABLE orders (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1,
    symbol_code TEXT NOT NULL,
    symbol_asset_class TEXT NOT NULL DEFAULT 'equity',
    order_type TEXT NOT NULL CHECK (order_type IN ('market', 'limit', 'stop', 'stop_limit')),
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity_amount DECIMAL(20, 8) NOT NULL CHECK (quantity_amount > 0),
    price DECIMAL(20, 8) NULL, -- NULL for market orders
    stop_price DECIMAL(20, 8) NULL,
    time_in_force TEXT NOT NULL DEFAULT 'DAY' CHECK (time_in_force IN ('DAY', 'GTC', 'IOC', 'FOK', 'GTD', 'ATO', 'ATC')),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'submitted', 'accepted', 'partially_filled', 'filled', 'cancelled', 'rejected', 'expired')),
    filled_quantity_amount DECIMAL(20, 8) NOT NULL DEFAULT 0.0,
    average_fill_price DECIMAL(20, 8) NULL,
    commission_amount DECIMAL(20, 8) NULL,
    commission_currency TEXT NULL,
    tags TEXT NOT NULL DEFAULT '[]', -- JSON array of tags
    strategy_id TEXT NULL,
    parent_order_id TEXT NULL,
    child_order_ids TEXT NOT NULL DEFAULT '[]', -- JSON array of child order IDs
    portfolio_id TEXT NULL,
    account_id TEXT NULL,
    FOREIGN KEY (parent_order_id) REFERENCES orders(id) ON DELETE SET NULL,
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE SET NULL,
    FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE SET NULL
);

-- Trades table
CREATE TABLE trades (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1,
    symbol_code TEXT NOT NULL,
    symbol_asset_class TEXT NOT NULL DEFAULT 'equity',
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity_amount DECIMAL(20, 8) NOT NULL CHECK (quantity_amount > 0),
    price DECIMAL(20, 8) NOT NULL CHECK (price > 0),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    order_id TEXT NOT NULL,
    position_id TEXT NULL,
    commission_amount DECIMAL(20, 8) NULL,
    commission_currency TEXT NULL,
    fees_amount DECIMAL(20, 8) NULL,
    fees_currency TEXT NULL,
    exchange TEXT NULL,
    strategy_id TEXT NULL,
    tags TEXT NOT NULL DEFAULT '[]', -- JSON array of tags
    portfolio_id TEXT NULL,
    account_id TEXT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE,
    FOREIGN KEY (position_id) REFERENCES positions(id) ON DELETE SET NULL,
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE SET NULL,
    FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE SET NULL
);

-- Positions table
CREATE TABLE positions (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1,
    symbol_code TEXT NOT NULL,
    symbol_asset_class TEXT NOT NULL DEFAULT 'equity',
    position_type TEXT NOT NULL CHECK (position_type IN ('long', 'short')),
    quantity_amount DECIMAL(20, 8) NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL CHECK (entry_price > 0),
    current_price DECIMAL(20, 8) NULL,
    entry_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    exit_timestamp TIMESTAMP NULL,
    entry_trades TEXT NOT NULL DEFAULT '[]', -- JSON array of trade IDs
    exit_trades TEXT NOT NULL DEFAULT '[]', -- JSON array of trade IDs
    strategy_id TEXT NULL,
    stop_loss DECIMAL(20, 8) NULL,
    take_profit DECIMAL(20, 8) NULL,
    tags TEXT NOT NULL DEFAULT '[]', -- JSON array of tags
    portfolio_id TEXT NOT NULL,
    account_id TEXT NULL,
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE,
    FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE SET NULL
);

-- Portfolio positions mapping table (many-to-many relationship helper)
CREATE TABLE portfolio_positions (
    portfolio_id TEXT NOT NULL,
    symbol_code TEXT NOT NULL,
    position_id TEXT NOT NULL,
    PRIMARY KEY (portfolio_id, symbol_code),
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE,
    FOREIGN KEY (position_id) REFERENCES positions(id) ON DELETE CASCADE
);

-- Audit logs table for all system actions
CREATE TABLE audit_logs (
    id TEXT PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT NULL, -- NULL for system-generated events
    action TEXT NOT NULL, -- e.g., 'order_created', 'trade_executed', 'position_opened'
    entity_type TEXT NOT NULL, -- e.g., 'order', 'trade', 'position', 'account', 'portfolio'
    entity_id TEXT NOT NULL,
    old_values TEXT NULL, -- JSON serialized old values for updates
    new_values TEXT NOT NULL, -- JSON serialized new values
    ip_address TEXT NULL,
    user_agent TEXT NULL,
    session_id TEXT NULL,
    strategy_id TEXT NULL, -- For strategy-generated actions
    metadata TEXT NOT NULL DEFAULT '{}' -- Additional metadata as JSON
);

-- Domain events table for event sourcing
CREATE TABLE domain_events (
    id TEXT PRIMARY KEY,
    event_id TEXT UNIQUE NOT NULL,
    occurred_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    aggregate_id TEXT NOT NULL,
    aggregate_type TEXT NOT NULL, -- e.g., 'Portfolio', 'Position'
    event_type TEXT NOT NULL, -- e.g., 'PositionOpened', 'OrderExecuted'
    event_data TEXT NOT NULL, -- JSON serialized event data
    processed BOOLEAN NOT NULL DEFAULT FALSE,
    processing_errors TEXT NULL, -- JSON array of processing errors
    version INTEGER NOT NULL DEFAULT 1
);

-- Indexes for performance optimization
CREATE INDEX idx_accounts_name ON accounts(name);
CREATE INDEX idx_accounts_active ON accounts(is_active);

CREATE INDEX idx_portfolios_account_id ON portfolios(account_id);
CREATE INDEX idx_portfolios_name ON portfolios(name);

CREATE INDEX idx_orders_symbol ON orders(symbol_code);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_created_at ON orders(created_at);
CREATE INDEX idx_orders_strategy_id ON orders(strategy_id);
CREATE INDEX idx_orders_portfolio_id ON orders(portfolio_id);

CREATE INDEX idx_trades_symbol ON trades(symbol_code);
CREATE INDEX idx_trades_timestamp ON trades(timestamp);
CREATE INDEX idx_trades_order_id ON trades(order_id);
CREATE INDEX idx_trades_position_id ON trades(position_id);
CREATE INDEX idx_trades_strategy_id ON trades(strategy_id);
CREATE INDEX idx_trades_portfolio_id ON trades(portfolio_id);

CREATE INDEX idx_positions_symbol ON positions(symbol_code);
CREATE INDEX idx_positions_entry_timestamp ON positions(entry_timestamp);
CREATE INDEX idx_positions_exit_timestamp ON positions(exit_timestamp);
CREATE INDEX idx_positions_portfolio_id ON positions(portfolio_id);
CREATE INDEX idx_positions_strategy_id ON positions(strategy_id);

CREATE INDEX idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_entity_type ON audit_logs(entity_type);
CREATE INDEX idx_audit_logs_entity_id ON audit_logs(entity_id);
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_strategy_id ON audit_logs(strategy_id);

CREATE INDEX idx_domain_events_aggregate_id ON domain_events(aggregate_id);
CREATE INDEX idx_domain_events_event_type ON domain_events(event_type);
CREATE INDEX idx_domain_events_occurred_at ON domain_events(occurred_at);
CREATE INDEX idx_domain_events_processed ON domain_events(processed);

-- Triggers for automatically updating updated_at timestamps
CREATE TRIGGER update_accounts_timestamp 
    AFTER UPDATE ON accounts
    FOR EACH ROW
    WHEN NEW.updated_at = OLD.updated_at
BEGIN
    UPDATE accounts SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER update_portfolios_timestamp 
    AFTER UPDATE ON portfolios
    FOR EACH ROW
    WHEN NEW.updated_at = OLD.updated_at
BEGIN
    UPDATE portfolios SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER update_orders_timestamp 
    AFTER UPDATE ON orders
    FOR EACH ROW
    WHEN NEW.updated_at = OLD.updated_at
BEGIN
    UPDATE orders SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER update_trades_timestamp 
    AFTER UPDATE ON trades
    FOR EACH ROW
    WHEN NEW.updated_at = OLD.updated_at
BEGIN
    UPDATE trades SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER update_positions_timestamp 
    AFTER UPDATE ON positions
    FOR EACH ROW
    WHEN NEW.updated_at = OLD.updated_at
BEGIN
    UPDATE positions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Views for common queries
CREATE VIEW v_open_positions AS
SELECT 
    p.*,
    (p.current_price - p.entry_price) * p.quantity_amount * 
    CASE WHEN p.position_type = 'long' THEN 1 ELSE -1 END as unrealized_pnl,
    ((p.current_price - p.entry_price) / p.entry_price) * 100 * 
    CASE WHEN p.position_type = 'long' THEN 1 ELSE -1 END as unrealized_pnl_percent
FROM positions p
WHERE p.exit_timestamp IS NULL;

CREATE VIEW v_active_orders AS
SELECT *
FROM orders
WHERE status IN ('pending', 'submitted', 'accepted', 'partially_filled');

CREATE VIEW v_portfolio_summary AS
SELECT 
    pf.id,
    pf.name,
    pf.account_id,
    pf.total_value_amount,
    pf.cash_balance_amount,
    COUNT(p.id) as position_count,
    SUM(CASE WHEN p.exit_timestamp IS NULL THEN 1 ELSE 0 END) as open_positions,
    pf.realized_pnl_amount + pf.unrealized_pnl_amount as total_pnl
FROM portfolios pf
LEFT JOIN positions p ON p.portfolio_id = pf.id
GROUP BY pf.id, pf.name, pf.account_id, pf.total_value_amount, pf.cash_balance_amount, pf.realized_pnl_amount, pf.unrealized_pnl_amount;

-- Constraints for data integrity
CREATE UNIQUE INDEX idx_portfolio_positions_unique ON portfolio_positions(portfolio_id, symbol_code);

-- Comments for documentation
-- This schema provides:
-- 1. Complete persistence for all trading entities
-- 2. Audit trail for all system actions
-- 3. Domain events for event sourcing
-- 4. Optimized indexes for performance
-- 5. Foreign key constraints for referential integrity
-- 6. Check constraints for data validation
-- 7. Automatic timestamp updates via triggers
-- 8. Convenient views for common queries