"""
Execution monitoring and performance tracking.

Provides comprehensive monitoring of execution quality, strategy performance,
and system health metrics.
"""

import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ..models.trading import Order, Portfolio, Trade
from ..models.types import Money, OrderStatus

logger = logging.getLogger(__name__)


class MonitorConfig(BaseModel):
    """Configuration for execution monitor retention and cleanup."""

    # Retention periods (number of items to keep)
    max_fill_times: int = Field(default=1000, description="Max fill time records")
    max_slippage_records: int = Field(default=1000, description="Max slippage records")
    max_trade_timestamps: int = Field(default=5000, description="Max trade timestamps")
    max_error_records: int = Field(default=1000, description="Max error records")
    max_portfolio_snapshots: int = Field(
        default=720, description="Max portfolio snapshots (12 hours)"
    )
    max_order_history: int = Field(default=10000, description="Max order history")

    # Cleanup periods (how often to run cleanup, in minutes)
    cleanup_interval_minutes: int = Field(default=60, description="Cleanup interval")
    order_retention_hours: int = Field(default=24, description="Order retention period")
    trade_retention_hours: int = Field(default=48, description="Trade retention period")
    error_retention_hours: int = Field(default=24, description="Error retention period")


class ExecutionMetrics(BaseModel):
    """Execution quality metrics."""

    total_orders: int = Field(default=0, description="Total orders processed")
    filled_orders: int = Field(default=0, description="Successfully filled orders")
    cancelled_orders: int = Field(default=0, description="Cancelled orders")
    rejected_orders: int = Field(default=0, description="Rejected orders")
    average_fill_time_seconds: Decimal = Field(
        default=Decimal("0"), description="Average fill time"
    )
    average_slippage_bps: Decimal = Field(
        default=Decimal("0"), description="Average slippage in basis points"
    )
    fill_rate_percent: Decimal = Field(
        default=Decimal("0"), description="Fill rate percentage"
    )
    commission_total: Money = Field(
        default_factory=lambda: Money(amount=Decimal("0"), currency="USD")
    )


class StrategyMetrics(BaseModel):
    """Strategy performance metrics."""

    strategy_id: str = Field(description="Strategy identifier")
    total_trades: int = Field(default=0, description="Total trades")
    winning_trades: int = Field(default=0, description="Winning trades")
    losing_trades: int = Field(default=0, description="Losing trades")
    total_pnl: Money = Field(
        default_factory=lambda: Money(amount=Decimal("0"), currency="USD")
    )
    win_rate_percent: Decimal = Field(
        default=Decimal("0"), description="Win rate percentage"
    )
    average_trade_pnl: Money = Field(
        default_factory=lambda: Money(amount=Decimal("0"), currency="USD")
    )
    max_drawdown: Decimal = Field(default=Decimal("0"), description="Maximum drawdown")
    sharpe_ratio: Optional[Decimal] = Field(default=None, description="Sharpe ratio")
    sortino_ratio: Optional[Decimal] = Field(default=None, description="Sortino ratio")
    profit_factor: Decimal = Field(default=Decimal("0"), description="Profit factor")


class SystemHealth(BaseModel):
    """System health metrics."""

    uptime_seconds: int = Field(description="System uptime in seconds")
    orders_per_minute: Decimal = Field(description="Order rate")
    trades_per_minute: Decimal = Field(description="Trade rate")
    error_rate_percent: Decimal = Field(description="Error rate percentage")
    memory_usage_mb: Optional[Decimal] = Field(default=None, description="Memory usage")
    cpu_usage_percent: Optional[Decimal] = Field(default=None, description="CPU usage")
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)


class ExecutionMonitor:
    """Comprehensive execution and performance monitoring with memory management."""

    def __init__(
        self, lookback_minutes: int = 60, config: Optional[MonitorConfig] = None
    ):
        """Initialize execution monitor with configurable retention policies."""
        self.lookback_minutes = lookback_minutes
        self.config = config or MonitorConfig()
        self.start_time = datetime.utcnow()
        self._last_cleanup = datetime.utcnow()

        # Order tracking with configurable limits
        self._orders: Dict[str, Order] = {}
        self._order_timestamps: Dict[str, datetime] = {}
        self._fill_times: deque = deque(maxlen=self.config.max_fill_times)
        self._slippage_records: deque = deque(maxlen=self.config.max_slippage_records)

        # Trade tracking with configurable limits
        self._trades: Dict[str, List[Trade]] = defaultdict(list)  # by strategy
        self._trade_timestamps: deque = deque(maxlen=self.config.max_trade_timestamps)

        # Error tracking with configurable limits
        self._errors: deque = deque(maxlen=self.config.max_error_records)
        self._error_timestamps: deque = deque(maxlen=self.config.max_error_records)

        # Performance tracking with configurable limits
        self._strategy_performance: Dict[str, StrategyMetrics] = {}
        self._portfolio_snapshots: deque = deque(
            maxlen=self.config.max_portfolio_snapshots
        )

        logger.info(
            f"ExecutionMonitor initialized with {lookback_minutes}min lookback, "
            f"cleanup every {self.config.cleanup_interval_minutes}min"
        )

    def record_order_submitted(self, order: Order) -> None:
        """Record order submission."""
        self._orders[order.id] = order
        self._order_timestamps[order.id] = datetime.utcnow()

        # Perform periodic cleanup to prevent memory leaks
        if self.should_cleanup():
            self.cleanup_old_data()

        logger.debug(f"Recorded order submission: {order.id}")

    def record_order_filled(
        self, order: Order, fill_time: Optional[datetime] = None
    ) -> None:
        """Record order fill."""
        if fill_time is None:
            fill_time = datetime.utcnow()

        # Calculate fill time
        if order.id in self._order_timestamps:
            submit_time = self._order_timestamps[order.id]
            fill_duration = (fill_time - submit_time).total_seconds()
            self._fill_times.append(fill_duration)

        # Calculate slippage if limit order
        if order.is_limit_order and order.price and order.average_fill_price:
            slippage_bps = (
                abs(order.average_fill_price - order.price) / order.price * 10000
            )
            self._slippage_records.append(slippage_bps)

        self._orders[order.id] = order

        logger.debug(f"Recorded order fill: {order.id}")

    def record_trade(self, trade: Trade) -> None:
        """Record executed trade."""
        strategy_id = trade.strategy_id or "default"
        self._trades[strategy_id].append(trade)
        self._trade_timestamps.append(datetime.utcnow())

        # Update strategy metrics
        self._update_strategy_metrics(strategy_id, trade)

        # Perform periodic cleanup to prevent memory leaks
        if self.should_cleanup():
            self.cleanup_old_data()

        logger.debug(f"Recorded trade: {trade.id} for strategy {strategy_id}")

    def record_error(self, error: str, context: Optional[Dict] = None) -> None:
        """Record system error."""
        self._errors.append(error)
        self._error_timestamps.append(datetime.utcnow())

        logger.warning(f"Recorded error: {error}")

    def record_portfolio_snapshot(self, portfolio: Portfolio) -> None:
        """Record portfolio snapshot for performance tracking."""
        self._portfolio_snapshots.append((datetime.utcnow(), portfolio))

        logger.debug(
            f"Recorded portfolio snapshot: value={portfolio.total_value.amount}"
        )

    def get_execution_metrics(self) -> ExecutionMetrics:
        """Get current execution metrics."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=self.lookback_minutes)

        # Filter recent orders
        recent_orders = [
            order
            for order_id, order in self._orders.items()
            if self._order_timestamps.get(order_id, datetime.min) >= cutoff_time
        ]

        total_orders = len(recent_orders)
        filled_orders = sum(1 for order in recent_orders if order.is_filled)
        cancelled_orders = sum(1 for order in recent_orders if order.is_cancelled)
        rejected_orders = sum(
            1 for order in recent_orders if order.status == OrderStatus.REJECTED
        )

        # Calculate averages
        recent_fill_times = [t for t in self._fill_times if t is not None]
        avg_fill_time = (
            Decimal(str(sum(recent_fill_times) / len(recent_fill_times)))
            if recent_fill_times
            else Decimal("0")
        )

        recent_slippage = list(self._slippage_records)
        avg_slippage = (
            Decimal(str(sum(recent_slippage) / len(recent_slippage)))
            if recent_slippage
            else Decimal("0")
        )

        fill_rate = (
            Decimal(str(filled_orders / total_orders * 100))
            if total_orders > 0
            else Decimal("0")
        )

        # Calculate total commission
        total_commission = Decimal("0")
        for order in recent_orders:
            if order.commission:
                total_commission += order.commission.amount

        return ExecutionMetrics(
            total_orders=total_orders,
            filled_orders=filled_orders,
            cancelled_orders=cancelled_orders,
            rejected_orders=rejected_orders,
            average_fill_time_seconds=avg_fill_time,
            average_slippage_bps=avg_slippage,
            fill_rate_percent=fill_rate,
            commission_total=Money(amount=total_commission, currency="USD"),
        )

    def get_strategy_metrics(self, strategy_id: str) -> Optional[StrategyMetrics]:
        """Get metrics for specific strategy."""
        return self._strategy_performance.get(strategy_id)

    def get_all_strategy_metrics(self) -> Dict[str, StrategyMetrics]:
        """Get metrics for all strategies."""
        return self._strategy_performance.copy()

    def get_system_health(self) -> SystemHealth:
        """Get system health metrics."""
        now = datetime.utcnow()
        uptime_seconds = int((now - self.start_time).total_seconds())

        # Calculate rates over last minute
        minute_ago = now - timedelta(minutes=1)
        recent_orders = sum(
            1
            for timestamp in self._order_timestamps.values()
            if timestamp >= minute_ago
        )
        recent_trades = sum(
            1 for timestamp in self._trade_timestamps if timestamp >= minute_ago
        )
        recent_errors = sum(
            1 for timestamp in self._error_timestamps if timestamp >= minute_ago
        )

        # Calculate error rate over last hour
        hour_ago = now - timedelta(hours=1)
        hour_orders = sum(
            1 for timestamp in self._order_timestamps.values() if timestamp >= hour_ago
        )
        hour_errors = sum(
            1 for timestamp in self._error_timestamps if timestamp >= hour_ago
        )
        error_rate = (
            Decimal(str(hour_errors / hour_orders * 100))
            if hour_orders > 0
            else Decimal("0")
        )

        return SystemHealth(
            uptime_seconds=uptime_seconds,
            orders_per_minute=Decimal(str(recent_orders)),
            trades_per_minute=Decimal(str(recent_trades)),
            error_rate_percent=error_rate,
            last_heartbeat=now,
        )

    def get_portfolio_performance(self, days: int = 30) -> Dict[str, Decimal]:
        """Get portfolio performance metrics over specified period."""
        if not self._portfolio_snapshots:
            return {}

        cutoff_time = datetime.utcnow() - timedelta(days=days)
        recent_snapshots = [
            (timestamp, portfolio)
            for timestamp, portfolio in self._portfolio_snapshots
            if timestamp >= cutoff_time
        ]

        if len(recent_snapshots) < 2:
            return {}

        # Get start and end values
        start_value = recent_snapshots[0][1].total_value.amount
        end_value = recent_snapshots[-1][1].total_value.amount

        # Calculate returns
        total_return = (
            (end_value - start_value) / start_value * 100
            if start_value > 0
            else Decimal("0")
        )

        # Calculate maximum drawdown
        high_water_mark = start_value
        max_drawdown = Decimal("0")

        for timestamp, portfolio in recent_snapshots:
            value = portfolio.total_value.amount
            if value > high_water_mark:
                high_water_mark = value

            drawdown = (
                (high_water_mark - value) / high_water_mark * 100
                if high_water_mark > 0
                else Decimal("0")
            )
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Calculate volatility (simplified)
        returns = []
        for i in range(1, len(recent_snapshots)):
            prev_value = recent_snapshots[i - 1][1].total_value.amount
            curr_value = recent_snapshots[i][1].total_value.amount
            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                returns.append(daily_return)

        if returns:
            avg_return = sum(returns) / len(returns)
            variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
            volatility = variance ** Decimal("0.5") * Decimal("252") ** Decimal(
                "0.5"
            )  # Annualized
        else:
            volatility = Decimal("0")

        # Calculate Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = Decimal("0.02")
        sharpe_ratio = (
            (total_return / 100 - risk_free_rate) / volatility
            if volatility > 0
            else Decimal("0")
        )

        return {
            "total_return_percent": total_return,
            "max_drawdown_percent": max_drawdown,
            "volatility_annualized": volatility * 100,
            "sharpe_ratio": sharpe_ratio,
            "start_value": start_value,
            "end_value": end_value,
        }

    def _update_strategy_metrics(self, strategy_id: str, trade: Trade) -> None:
        """Update strategy performance metrics."""
        if strategy_id not in self._strategy_performance:
            self._strategy_performance[strategy_id] = StrategyMetrics(
                strategy_id=strategy_id
            )

        metrics = self._strategy_performance[strategy_id]
        strategy_trades = self._trades[strategy_id]

        # Update basic counts
        metrics.total_trades = len(strategy_trades)

        # Calculate P&L (simplified - would need more sophisticated calculation)
        total_pnl = Decimal("0")
        winning_trades = 0
        losing_trades = 0

        # Group trades by symbol to calculate realized P&L
        symbol_trades = defaultdict(list)
        for t in strategy_trades:
            symbol_trades[t.symbol.code].append(t)

        for symbol, trades in symbol_trades.items():
            trades.sort(key=lambda x: x.timestamp)
            position_size = Decimal("0")
            cost_basis = Decimal("0")

            for trade in trades:
                if trade.is_buy:
                    # Add to position
                    new_size = position_size + trade.quantity.amount
                    if new_size != 0:
                        cost_basis = (
                            position_size * cost_basis + trade.total_cost
                        ) / new_size
                    position_size = new_size
                else:
                    # Sell from position
                    if position_size > 0:
                        sold_quantity = min(trade.quantity.amount, position_size)
                        trade_pnl = (trade.price - cost_basis) * sold_quantity
                        total_pnl += trade_pnl

                        if trade_pnl > 0:
                            winning_trades += 1
                        else:
                            losing_trades += 1

                        position_size -= sold_quantity

        # Update metrics
        metrics.total_pnl = Money(amount=total_pnl, currency="USD")
        metrics.winning_trades = winning_trades
        metrics.losing_trades = losing_trades

        if metrics.total_trades > 0:
            metrics.win_rate_percent = Decimal(
                str(winning_trades / metrics.total_trades * 100)
            )
            metrics.average_trade_pnl = Money(
                amount=total_pnl / metrics.total_trades, currency="USD"
            )

        # Calculate profit factor
        gross_profit = sum(
            (t.price - t.price * Decimal("0.9")) * t.quantity.amount  # Simplified
            for t in strategy_trades
            if t.is_sell
        )
        gross_loss = sum(
            (t.price * Decimal("1.1") - t.price) * t.quantity.amount  # Simplified
            for t in strategy_trades
            if t.is_sell
        )

        if gross_loss != 0:
            metrics.profit_factor = gross_profit / abs(gross_loss)

        self._strategy_performance[strategy_id] = metrics

    def get_recent_errors(self, limit: int = 10) -> List[Tuple[datetime, str]]:
        """Get recent errors with timestamps."""
        recent = list(zip(self._error_timestamps, self._errors))[-limit:]
        return recent

    def cleanup_old_data(self) -> Dict[str, int]:
        """
        Clean up old data beyond retention periods to prevent memory leaks.

        Returns:
            Dict with counts of cleaned up items by category
        """
        now = datetime.utcnow()
        cleanup_stats = {
            "orders_cleaned": 0,
            "trades_cleaned": 0,
            "errors_cleaned": 0,
            "timestamps_cleaned": 0,
        }

        # Clean up old orders beyond retention period
        order_cutoff = now - timedelta(hours=self.config.order_retention_hours)
        orders_to_remove = []
        for order_id, timestamp in self._order_timestamps.items():
            if timestamp < order_cutoff:
                orders_to_remove.append(order_id)

        for order_id in orders_to_remove:
            self._orders.pop(order_id, None)
            self._order_timestamps.pop(order_id, None)
            cleanup_stats["orders_cleaned"] += 1

        # Clean up old trades beyond retention period
        trade_cutoff = now - timedelta(hours=self.config.trade_retention_hours)
        original_trade_count = len(self._trade_timestamps)

        # Filter trade timestamps
        self._trade_timestamps = deque(
            (ts for ts in self._trade_timestamps if ts > trade_cutoff),
            maxlen=self.config.max_trade_timestamps,
        )
        cleanup_stats["trades_cleaned"] = original_trade_count - len(
            self._trade_timestamps
        )

        # Clean up old errors beyond retention period
        error_cutoff = now - timedelta(hours=self.config.error_retention_hours)
        original_error_count = len(self._error_timestamps)

        # Filter errors and timestamps together
        filtered_errors = []
        filtered_timestamps = []
        for error, timestamp in zip(self._errors, self._error_timestamps):
            if timestamp > error_cutoff:
                filtered_errors.append(error)
                filtered_timestamps.append(timestamp)

        self._errors = deque(filtered_errors, maxlen=self.config.max_error_records)
        self._error_timestamps = deque(
            filtered_timestamps, maxlen=self.config.max_error_records
        )
        cleanup_stats["errors_cleaned"] = original_error_count - len(self._errors)

        # Update last cleanup time
        self._last_cleanup = now

        if any(cleanup_stats.values()):
            logger.info(f"Memory cleanup completed: {cleanup_stats}")

        return cleanup_stats

    def should_cleanup(self) -> bool:
        """Check if cleanup should be performed based on configured interval."""
        cleanup_interval = timedelta(minutes=self.config.cleanup_interval_minutes)
        return datetime.utcnow() - self._last_cleanup >= cleanup_interval

    def get_memory_stats(self) -> Dict[str, int]:
        """Get current memory usage statistics."""
        return {
            "orders_count": len(self._orders),
            "order_timestamps_count": len(self._order_timestamps),
            "fill_times_count": len(self._fill_times),
            "slippage_records_count": len(self._slippage_records),
            "trade_timestamps_count": len(self._trade_timestamps),
            "error_records_count": len(self._errors),
            "error_timestamps_count": len(self._error_timestamps),
            "portfolio_snapshots_count": len(self._portfolio_snapshots),
            "strategy_performance_count": len(self._strategy_performance),
            "total_trades_count": sum(len(trades) for trades in self._trades.values()),
        }

    def reset_metrics(self) -> None:
        """Reset all metrics (useful for testing)."""
        self._orders.clear()
        self._order_timestamps.clear()
        self._fill_times.clear()
        self._slippage_records.clear()
        self._trades.clear()
        self._trade_timestamps.clear()
        self._errors.clear()
        self._error_timestamps.clear()
        self._strategy_performance.clear()
        self._portfolio_snapshots.clear()

        self.start_time = datetime.utcnow()
        logger.info("ExecutionMonitor metrics reset")
