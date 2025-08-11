"""
Trading service with persistent storage integration.

This module provides a service layer that integrates the trading operations
with persistent storage, handling automatic saving of trades, positions,
accounts, and portfolios with audit logging.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from ..models.trading import Account, Order, Portfolio, Position, Trade
from ..models.types import Money, OrderSide
from .database import DatabaseManager, get_database
from .repositories import (
    AccountRepository,
    AuditLogRepository,
    OrderRepository,
    PortfolioRepository,
    PositionRepository,
    TradeRepository,
)

logger = logging.getLogger(__name__)


class PersistentTradingService:
    """
    Service layer for trading operations with persistent storage.

    This service handles all trading operations while automatically
    persisting entities and maintaining audit trails.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize trading service with database repositories.

        Args:
            db_manager: Database manager instance (will create default if None)
        """
        self.db_manager = db_manager
        self._repositories_initialized = False

    async def _ensure_repositories(self) -> None:
        """Ensure repositories are initialized."""
        if not self._repositories_initialized:
            if self.db_manager is None:
                self.db_manager = await get_database()

            self.account_repo = AccountRepository(self.db_manager)
            self.portfolio_repo = PortfolioRepository(self.db_manager)
            self.order_repo = OrderRepository(self.db_manager)
            self.trade_repo = TradeRepository(self.db_manager)
            self.position_repo = PositionRepository(self.db_manager)
            self.audit_repo = AuditLogRepository(self.db_manager)

            self._repositories_initialized = True

    # Account operations
    async def create_account(self, account: Account, user_id: Optional[str] = None) -> Account:
        """
        Create a new trading account with audit logging.

        Args:
            account: Account to create
            user_id: ID of the user creating the account

        Returns:
            Saved account entity
        """
        await self._ensure_repositories()

        # Save account
        saved_account = await self.account_repo.save(account)

        # Create audit log
        await self.audit_repo.create_log(
            action="account_created",
            entity_type="account",
            entity_id=saved_account.id,
            new_values=saved_account.model_dump(mode="json"),
            user_id=user_id,
        )

        logger.info(f"Created account {saved_account.name} (ID: {saved_account.id})")
        return saved_account

    async def update_account_balances(
        self,
        account_id: str,
        cash_balance: Money,
        total_equity: Money,
        buying_power: Money,
        maintenance_margin: Money,
        user_id: Optional[str] = None,
    ) -> Optional[Account]:
        """
        Update account balances with audit logging.

        Args:
            account_id: ID of the account to update
            cash_balance: New cash balance
            total_equity: New total equity
            buying_power: New buying power
            maintenance_margin: New maintenance margin
            user_id: ID of the user updating the account

        Returns:
            Updated account entity or None if not found
        """
        await self._ensure_repositories()

        # Get existing account
        account = await self.account_repo.find_by_id(account_id)
        if not account:
            logger.warning(f"Account not found: {account_id}")
            return None

        # Store old values for audit
        old_values = account.model_dump(mode="json")

        # Update balances
        updated_account = account.update_balances(
            cash_balance=cash_balance,
            total_equity=total_equity,
            buying_power=buying_power,
            maintenance_margin=maintenance_margin,
        )

        # Save updated account
        saved_account = await self.account_repo.save(updated_account)

        # Create audit log
        await self.audit_repo.create_log(
            action="account_balances_updated",
            entity_type="account",
            entity_id=saved_account.id,
            old_values=old_values,
            new_values=saved_account.model_dump(mode="json"),
            user_id=user_id,
        )

        logger.info(f"Updated balances for account {saved_account.name}")
        return saved_account

    # Portfolio operations
    async def create_portfolio(
        self, portfolio: Portfolio, user_id: Optional[str] = None
    ) -> Portfolio:
        """
        Create a new portfolio with audit logging.

        Args:
            portfolio: Portfolio to create
            user_id: ID of the user creating the portfolio

        Returns:
            Saved portfolio entity
        """
        await self._ensure_repositories()

        # Save portfolio
        saved_portfolio = await self.portfolio_repo.save(portfolio)

        # Create audit log
        await self.audit_repo.create_log(
            action="portfolio_created",
            entity_type="portfolio",
            entity_id=saved_portfolio.id,
            new_values=saved_portfolio.model_dump(mode="json"),
            user_id=user_id,
        )

        logger.info(f"Created portfolio {saved_portfolio.name} (ID: {saved_portfolio.id})")
        return saved_portfolio

    async def update_portfolio_metrics(
        self,
        portfolio_id: str,
        unrealized_pnl: Money,
        total_value: Money,
        user_id: Optional[str] = None,
    ) -> Optional[Portfolio]:
        """
        Update portfolio metrics with audit logging.

        Args:
            portfolio_id: ID of the portfolio to update
            unrealized_pnl: New unrealized P&L
            total_value: New total value
            user_id: ID of the user updating the portfolio

        Returns:
            Updated portfolio entity or None if not found
        """
        await self._ensure_repositories()

        # Get existing portfolio
        portfolio = await self.portfolio_repo.find_by_id(portfolio_id)
        if not portfolio:
            logger.warning(f"Portfolio not found: {portfolio_id}")
            return None

        # Store old values for audit
        old_values = portfolio.model_dump(mode="json")

        # Update metrics
        updated_portfolio = portfolio.update_metrics(
            unrealized_pnl=unrealized_pnl,
            total_value=total_value,
        )

        # Save updated portfolio
        saved_portfolio = await self.portfolio_repo.save(updated_portfolio)

        # Create audit log
        await self.audit_repo.create_log(
            action="portfolio_metrics_updated",
            entity_type="portfolio",
            entity_id=saved_portfolio.id,
            old_values=old_values,
            new_values=saved_portfolio.model_dump(mode="json"),
            user_id=user_id,
        )

        logger.info(f"Updated metrics for portfolio {saved_portfolio.name}")
        return saved_portfolio

    # Order operations
    async def create_order(
        self, order: Order, strategy_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> Order:
        """
        Create a new order with audit logging.

        Args:
            order: Order to create
            strategy_id: ID of the strategy creating the order
            user_id: ID of the user creating the order

        Returns:
            Saved order entity
        """
        await self._ensure_repositories()

        # Add strategy ID if provided
        if strategy_id and not order.strategy_id:
            order = order.update(strategy_id=strategy_id)

        # Save order
        saved_order = await self.order_repo.save(order)

        # Create audit log
        await self.audit_repo.create_log(
            action="order_created",
            entity_type="order",
            entity_id=saved_order.id,
            new_values=saved_order.model_dump(mode="json"),
            user_id=user_id,
            strategy_id=strategy_id,
            metadata={
                "symbol": saved_order.symbol.code,
                "side": saved_order.side.value,
                "quantity": float(saved_order.quantity.amount),
                "order_type": saved_order.order_type.value,
            },
        )

        logger.info(
            f"Created {saved_order.order_type.value} {saved_order.side.value} order "
            f"for {saved_order.quantity.amount} {saved_order.symbol.code}"
        )
        return saved_order

    async def update_order(
        self, order: Order, strategy_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> Order:
        """
        Update an existing order with audit logging.

        Args:
            order: Updated order entity
            strategy_id: ID of the strategy updating the order
            user_id: ID of the user updating the order

        Returns:
            Saved order entity
        """
        await self._ensure_repositories()

        # Get existing order for audit trail
        existing_order = await self.order_repo.find_by_id(order.id)
        old_values = existing_order.model_dump(mode="json") if existing_order else {}

        # Save updated order
        saved_order = await self.order_repo.save(order)

        # Create audit log
        await self.audit_repo.create_log(
            action="order_updated",
            entity_type="order",
            entity_id=saved_order.id,
            old_values=old_values,
            new_values=saved_order.model_dump(mode="json"),
            user_id=user_id,
            strategy_id=strategy_id,
            metadata={
                "status_change": f"{existing_order.status.value if existing_order else 'unknown'} -> {saved_order.status.value}",
                "symbol": saved_order.symbol.code,
            },
        )

        logger.info(f"Updated order {saved_order.id} - Status: {saved_order.status.value}")
        return saved_order

    # Trade operations
    async def create_trade(
        self, trade: Trade, strategy_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> Trade:
        """
        Create a new trade with audit logging.

        Args:
            trade: Trade to create
            strategy_id: ID of the strategy that executed the trade
            user_id: ID of the user (if manually executed)

        Returns:
            Saved trade entity
        """
        await self._ensure_repositories()

        # Add strategy ID if provided
        if strategy_id and not trade.strategy_id:
            trade = trade.update(strategy_id=strategy_id)

        # Save trade
        saved_trade = await self.trade_repo.save(trade)

        # Create audit log
        await self.audit_repo.create_log(
            action="trade_executed",
            entity_type="trade",
            entity_id=saved_trade.id,
            new_values=saved_trade.model_dump(mode="json"),
            user_id=user_id,
            strategy_id=strategy_id,
            metadata={
                "symbol": saved_trade.symbol.code,
                "side": saved_trade.side.value,
                "quantity": float(saved_trade.quantity.amount),
                "price": float(saved_trade.price),
                "notional_value": float(saved_trade.notional_value),
            },
        )

        logger.info(
            f"Executed {saved_trade.side.value} trade: "
            f"{saved_trade.quantity.amount} {saved_trade.symbol.code} @ ${saved_trade.price}"
        )
        return saved_trade

    # Position operations
    async def create_position(
        self, position: Position, strategy_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> Position:
        """
        Create a new position with audit logging.

        Args:
            position: Position to create
            strategy_id: ID of the strategy that opened the position
            user_id: ID of the user (if manually opened)

        Returns:
            Saved position entity
        """
        await self._ensure_repositories()

        # Add strategy ID if provided
        if strategy_id and not position.strategy_id:
            position = position.update(strategy_id=strategy_id)

        # Save position
        saved_position = await self.position_repo.save(position)

        # Create audit log
        await self.audit_repo.create_log(
            action="position_opened",
            entity_type="position",
            entity_id=saved_position.id,
            new_values=saved_position.model_dump(mode="json"),
            user_id=user_id,
            strategy_id=strategy_id,
            metadata={
                "symbol": saved_position.symbol.code,
                "position_type": saved_position.position_type.value,
                "quantity": float(saved_position.quantity.amount),
                "entry_price": float(saved_position.entry_price),
                "cost_basis": float(saved_position.cost_basis),
            },
        )

        logger.info(
            f"Opened {saved_position.position_type.value} position: "
            f"{saved_position.quantity.amount} {saved_position.symbol.code} @ ${saved_position.entry_price}"
        )
        return saved_position

    async def update_position(
        self, position: Position, strategy_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> Position:
        """
        Update an existing position with audit logging.

        Args:
            position: Updated position entity
            strategy_id: ID of the strategy updating the position
            user_id: ID of the user updating the position

        Returns:
            Saved position entity
        """
        await self._ensure_repositories()

        # Get existing position for audit trail
        existing_position = await self.position_repo.find_by_id(position.id)
        old_values = existing_position.model_dump(mode="json") if existing_position else {}

        # Save updated position
        saved_position = await self.position_repo.save(position)

        # Determine action type
        action = "position_updated"
        if saved_position.is_closed and (not existing_position or existing_position.is_open):
            action = "position_closed"

        # Create audit log
        await self.audit_repo.create_log(
            action=action,
            entity_type="position",
            entity_id=saved_position.id,
            old_values=old_values,
            new_values=saved_position.model_dump(mode="json"),
            user_id=user_id,
            strategy_id=strategy_id,
            metadata={
                "symbol": saved_position.symbol.code,
                "unrealized_pnl": float(saved_position.unrealized_pnl or 0),
                "is_closed": saved_position.is_closed,
            },
        )

        logger.info(
            f"Updated position {saved_position.id} - Status: {'closed' if saved_position.is_closed else 'open'}"
        )
        return saved_position

    # Query operations
    async def get_account(self, account_id: str) -> Optional[Account]:
        """Get account by ID."""
        await self._ensure_repositories()
        return await self.account_repo.find_by_id(account_id)

    async def get_active_accounts(self) -> List[Account]:
        """Get all active accounts."""
        await self._ensure_repositories()
        return await self.account_repo.find_active_accounts()

    async def get_portfolio(self, portfolio_id: str) -> Optional[Portfolio]:
        """Get portfolio by ID."""
        await self._ensure_repositories()
        return await self.portfolio_repo.find_by_id(portfolio_id)

    async def get_portfolios_by_account(self, account_id: str) -> List[Portfolio]:
        """Get portfolios by account ID."""
        await self._ensure_repositories()
        return await self.portfolio_repo.find_by_account_id(account_id)

    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        await self._ensure_repositories()
        return await self.order_repo.find_by_id(order_id)

    async def get_active_orders(self) -> List[Order]:
        """Get all active orders."""
        await self._ensure_repositories()
        return await self.order_repo.find_active_orders()

    async def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get orders by symbol."""
        await self._ensure_repositories()
        return await self.order_repo.find_by_symbol(symbol)

    async def get_trade(self, trade_id: str) -> Optional[Trade]:
        """Get trade by ID."""
        await self._ensure_repositories()
        return await self.trade_repo.find_by_id(trade_id)

    async def get_trades_by_symbol(self, symbol: str) -> List[Trade]:
        """Get trades by symbol."""
        await self._ensure_repositories()
        return await self.trade_repo.find_by_symbol(symbol)

    async def get_trades_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[Trade]:
        """Get trades within date range."""
        await self._ensure_repositories()
        return await self.trade_repo.find_by_date_range(start_date, end_date)

    async def get_position(self, position_id: str) -> Optional[Position]:
        """Get position by ID."""
        await self._ensure_repositories()
        return await self.position_repo.find_by_id(position_id)

    async def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        await self._ensure_repositories()
        return await self.position_repo.find_open_positions()

    async def get_positions_by_portfolio(self, portfolio_id: str) -> List[Position]:
        """Get positions by portfolio ID."""
        await self._ensure_repositories()
        return await self.position_repo.find_by_portfolio_id(portfolio_id)

    async def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """Get positions by symbol."""
        await self._ensure_repositories()
        return await self.position_repo.find_by_symbol(symbol)

    # Audit operations
    async def get_audit_logs_for_entity(
        self, entity_type: str, entity_id: str, limit: Optional[int] = 100
    ) -> List[Dict]:
        """Get audit logs for a specific entity."""
        await self._ensure_repositories()
        return await self.audit_repo.find_by_entity(entity_type, entity_id, limit)

    async def get_audit_logs_by_action(
        self,
        action: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = 100,
    ) -> List[Dict]:
        """Get audit logs by action type."""
        await self._ensure_repositories()
        return await self.audit_repo.find_by_action(action, start_date, end_date, limit)

    # Statistics and reporting
    async def get_portfolio_summary(self, portfolio_id: str) -> Optional[Dict]:
        """Get portfolio summary with positions and metrics."""
        await self._ensure_repositories()

        portfolio = await self.get_portfolio(portfolio_id)
        if not portfolio:
            return None

        positions = await self.get_positions_by_portfolio(portfolio_id)

        return {
            "portfolio": portfolio.model_dump(mode="json"),
            "positions": [pos.model_dump(mode="json") for pos in positions],
            "open_position_count": len([p for p in positions if p.is_open]),
            "closed_position_count": len([p for p in positions if p.is_closed]),
            "total_positions": len(positions),
        }

    async def get_trading_statistics(self, start_date: datetime, end_date: datetime) -> Dict:
        """Get trading statistics for a date range."""
        await self._ensure_repositories()

        trades = await self.get_trades_by_date_range(start_date, end_date)

        if not trades:
            return {
                "total_trades": 0,
                "buy_trades": 0,
                "sell_trades": 0,
                "total_volume": 0.0,
                "total_notional": 0.0,
                "average_trade_size": 0.0,
            }

        buy_trades = [t for t in trades if t.side == OrderSide.BUY]
        sell_trades = [t for t in trades if t.side == OrderSide.SELL]

        total_volume = sum(float(t.quantity.amount) for t in trades)
        total_notional = sum(float(t.notional_value) for t in trades)

        return {
            "total_trades": len(trades),
            "buy_trades": len(buy_trades),
            "sell_trades": len(sell_trades),
            "total_volume": total_volume,
            "total_notional": total_notional,
            "average_trade_size": total_volume / len(trades) if trades else 0.0,
            "average_notional": total_notional / len(trades) if trades else 0.0,
        }


# Global service instance
_trading_service: Optional[PersistentTradingService] = None


async def get_trading_service() -> PersistentTradingService:
    """
    Get the global trading service instance.

    Returns:
        Initialized trading service
    """
    global _trading_service
    if _trading_service is None:
        _trading_service = PersistentTradingService()

    return _trading_service
