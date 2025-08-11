-- Simple schema for testing
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS accounts (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    name TEXT NOT NULL,
    account_type TEXT NOT NULL,
    base_currency TEXT NOT NULL DEFAULT 'USD',
    cash_balance_amount REAL NOT NULL DEFAULT 0.0,
    cash_balance_currency TEXT NOT NULL DEFAULT 'USD',
    total_equity_amount REAL NOT NULL DEFAULT 0.0,
    total_equity_currency TEXT NOT NULL DEFAULT 'USD',
    buying_power_amount REAL NOT NULL DEFAULT 0.0,
    buying_power_currency TEXT NOT NULL DEFAULT 'USD',
    maintenance_margin_amount REAL NOT NULL DEFAULT 0.0,
    maintenance_margin_currency TEXT NOT NULL DEFAULT 'USD',
    max_position_size REAL NOT NULL DEFAULT 10.0,
    max_portfolio_risk REAL NOT NULL DEFAULT 2.0,
    is_active INTEGER NOT NULL DEFAULT 1,
    risk_parameters TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS portfolios (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    name TEXT NOT NULL,
    account_id TEXT NOT NULL,
    base_currency TEXT NOT NULL DEFAULT 'USD',
    cash_balance_amount REAL NOT NULL DEFAULT 0.0,
    cash_balance_currency TEXT NOT NULL DEFAULT 'USD',
    total_value_amount REAL NOT NULL DEFAULT 0.0,
    total_value_currency TEXT NOT NULL DEFAULT 'USD',
    unrealized_pnl_amount REAL NOT NULL DEFAULT 0.0,
    unrealized_pnl_currency TEXT NOT NULL DEFAULT 'USD',
    realized_pnl_amount REAL NOT NULL DEFAULT 0.0,
    realized_pnl_currency TEXT NOT NULL DEFAULT 'USD',
    daily_pnl_amount REAL NOT NULL DEFAULT 0.0,
    daily_pnl_currency TEXT NOT NULL DEFAULT 'USD',
    max_drawdown REAL NOT NULL DEFAULT 0.0,
    risk_metrics TEXT NOT NULL DEFAULT '{}',
    performance_metrics TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    symbol_code TEXT NOT NULL,
    symbol_asset_class TEXT NOT NULL DEFAULT 'equity',
    order_type TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity_amount REAL NOT NULL,
    price REAL,
    time_in_force TEXT NOT NULL DEFAULT 'DAY',
    status TEXT NOT NULL DEFAULT 'pending',
    filled_quantity_amount REAL NOT NULL DEFAULT 0.0,
    tags TEXT NOT NULL DEFAULT '[]',
    child_order_ids TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    symbol_code TEXT NOT NULL,
    symbol_asset_class TEXT NOT NULL DEFAULT 'equity',
    side TEXT NOT NULL,
    quantity_amount REAL NOT NULL,
    price REAL NOT NULL,
    timestamp TEXT NOT NULL,
    order_id TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS positions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    symbol_code TEXT NOT NULL,
    symbol_asset_class TEXT NOT NULL DEFAULT 'equity',
    position_type TEXT NOT NULL,
    quantity_amount REAL NOT NULL,
    entry_price REAL NOT NULL,
    entry_timestamp TEXT NOT NULL,
    entry_trades TEXT NOT NULL DEFAULT '[]',
    exit_trades TEXT NOT NULL DEFAULT '[]',
    tags TEXT NOT NULL DEFAULT '[]',
    portfolio_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    action TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    old_values TEXT,
    new_values TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);
