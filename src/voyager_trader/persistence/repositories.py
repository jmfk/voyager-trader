"""
Repository implementations for VOYAGER Trader entities.

This module provides concrete repository implementations for all trading
entities including accounts, portfolios, orders, trades, and positions.
"""

import json
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from ..models.base import BaseEntity, Repository
from ..models.trading import Account, Order, Portfolio, Position, Trade
from ..models.types import (
    Currency,
    Money,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionType,
    Quantity,
    Symbol,
)
from .database import DatabaseManager

T = TypeVar("T", bound=BaseEntity)


class BaseRepository(Repository, Generic[T]):
    """
    Base repository implementation with common CRUD operations.

    Provides shared functionality for all entity repositories including
    serialization, deserialization, and basic database operations.
    """

    def __init__(
        self, db_manager: DatabaseManager, entity_class: Type[T], table_name: str
    ):
        """
        Initialize base repository.

        Args:
            db_manager: Database manager instance
            entity_class: Entity class this repository manages
            table_name: Database table name
        """
        self.db = db_manager
        self.entity_class = entity_class
        self.table_name = table_name

    def _serialize_entity(self, entity: T) -> Dict[str, Any]:
        """
        Serialize entity to database format.

        Args:
            entity: Entity to serialize

        Returns:
            Dictionary with serialized entity data
        """
        data = entity.model_dump()

        # Convert datetime objects to ISO format
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, Decimal):
                data[key] = float(value)
            elif isinstance(value, (list, dict)):
                data[key] = json.dumps(value, default=str)

        return data

    def _deserialize_entity(self, data: Dict[str, Any]) -> T:
        """
        Deserialize database data to entity.

        Args:
            data: Database row data

        Returns:
            Deserialized entity instance
        """
        # Convert ISO datetime strings back to datetime objects
        for key, value in data.items():
            if key.endswith("_at") and isinstance(value, str):
                data[key] = datetime.fromisoformat(value.replace("Z", "+00:00"))
            elif key.endswith("_ids") and isinstance(value, str):
                data[key] = json.loads(value) if value else []
            elif key in (
                "tags",
                "risk_parameters",
                "performance_metrics",
                "risk_metrics",
            ):
                data[key] = (
                    json.loads(value)
                    if value
                    else ({} if "parameters" in key or "metrics" in key else [])
                )

        return self.entity_class.model_validate(data)

    async def find_by_id(self, entity_id: str) -> Optional[T]:
        """Find entity by ID."""
        query = f"SELECT * FROM {self.table_name} WHERE id = ?"
        result = await self.db.execute(query, (entity_id,), fetch="one")

        if result:
            column_names = (
                [description[0] for description in result.keys()]
                if hasattr(result, "keys")
                else []
            )
            if not column_names:
                # Fallback: get column names from table info
                table_info = await self.db.get_table_info(self.table_name)
                column_names = [col["name"] for col in table_info]

            data = dict(zip(column_names, result))
            return self._deserialize_entity(data)

        return None

    async def save(self, entity: T) -> T:
        """Save entity to database."""
        data = self._serialize_entity(entity)

        # Check if entity exists
        existing = await self.find_by_id(entity.id)

        if existing:
            # Update existing entity
            set_clause = ", ".join([f"{key} = ?" for key in data.keys() if key != "id"])
            query = f"UPDATE {self.table_name} SET {set_clause} WHERE id = ?"

            values = [data[key] for key in data.keys() if key != "id"] + [entity.id]
            await self.db.execute(query, tuple(values))
        else:
            # Insert new entity
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?" for _ in data])
            query = f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})"

            await self.db.execute(query, tuple(data.values()))

        return entity

    async def delete(self, entity_id: str) -> bool:
        """Delete entity by ID."""
        query = f"DELETE FROM {self.table_name} WHERE id = ?"
        result = await self.db.execute(query, (entity_id,))
        return result > 0

    async def find_all(self, limit: Optional[int] = None, offset: int = 0) -> List[T]:
        """Find all entities with optional pagination."""
        query = f"SELECT * FROM {self.table_name} ORDER BY created_at DESC"

        if limit:
            query += f" LIMIT {limit} OFFSET {offset}"

        results = await self.db.execute(query, fetch="all")
        entities = []

        if results:
            # Get column names
            table_info = await self.db.get_table_info(self.table_name)
            column_names = [col["name"] for col in table_info]

            for row in results:
                data = dict(zip(column_names, row))
                entities.append(self._deserialize_entity(data))

        return entities


class AccountRepository(BaseRepository[Account]):
    """Repository for Account entities."""

    def __init__(self, db_manager: DatabaseManager):
        super().__init__(db_manager, Account, "accounts")

    def _serialize_entity(self, entity: Account) -> Dict[str, Any]:
        """Serialize Account entity for database storage."""
        data = {
            "id": entity.id,
            "created_at": entity.created_at.isoformat(),
            "updated_at": entity.updated_at.isoformat(),
            "version": entity.version,
            "name": entity.name,
            "account_type": entity.account_type,
            "base_currency": entity.base_currency.value,
            "cash_balance_amount": float(entity.cash_balance.amount),
            "cash_balance_currency": entity.cash_balance.currency.value,
            "total_equity_amount": float(entity.total_equity.amount),
            "total_equity_currency": entity.total_equity.currency.value,
            "buying_power_amount": float(entity.buying_power.amount),
            "buying_power_currency": entity.buying_power.currency.value,
            "maintenance_margin_amount": float(entity.maintenance_margin.amount),
            "maintenance_margin_currency": entity.maintenance_margin.currency.value,
            "max_position_size": float(entity.max_position_size),
            "max_portfolio_risk": float(entity.max_portfolio_risk),
            "is_active": entity.is_active,
            "risk_parameters": json.dumps(entity.risk_parameters),
        }

        # Handle optional fields
        if entity.day_trading_buying_power:
            data["day_trading_buying_power_amount"] = float(
                entity.day_trading_buying_power.amount
            )
            data[
                "day_trading_buying_power_currency"
            ] = entity.day_trading_buying_power.currency.value

        if entity.daily_loss_limit:
            data["daily_loss_limit_amount"] = float(entity.daily_loss_limit.amount)
            data["daily_loss_limit_currency"] = entity.daily_loss_limit.currency.value

        return data

    def _deserialize_entity(self, data: Dict[str, Any]) -> Account:
        """Deserialize database data to Account entity."""
        # Convert timestamp strings
        if isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(
                data["created_at"].replace("Z", "+00:00")
            )
        if isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(
                data["updated_at"].replace("Z", "+00:00")
            )

        # Convert Money fields
        data["base_currency"] = Currency(data["base_currency"])
        data["cash_balance"] = Money(
            amount=Decimal(str(data["cash_balance_amount"])),
            currency=Currency(data["cash_balance_currency"]),
        )
        data["total_equity"] = Money(
            amount=Decimal(str(data["total_equity_amount"])),
            currency=Currency(data["total_equity_currency"]),
        )
        data["buying_power"] = Money(
            amount=Decimal(str(data["buying_power_amount"])),
            currency=Currency(data["buying_power_currency"]),
        )
        data["maintenance_margin"] = Money(
            amount=Decimal(str(data["maintenance_margin_amount"])),
            currency=Currency(data["maintenance_margin_currency"]),
        )

        # Handle optional Money fields
        if data.get("day_trading_buying_power_amount") is not None:
            data["day_trading_buying_power"] = Money(
                amount=Decimal(str(data["day_trading_buying_power_amount"])),
                currency=Currency(data["day_trading_buying_power_currency"]),
            )

        if data.get("daily_loss_limit_amount") is not None:
            data["daily_loss_limit"] = Money(
                amount=Decimal(str(data["daily_loss_limit_amount"])),
                currency=Currency(data["daily_loss_limit_currency"]),
            )

        # Convert Decimal fields
        data["max_position_size"] = Decimal(str(data["max_position_size"]))
        data["max_portfolio_risk"] = Decimal(str(data["max_portfolio_risk"]))

        # Parse JSON fields
        data["risk_parameters"] = (
            json.loads(data["risk_parameters"]) if data["risk_parameters"] else {}
        )

        # Remove database-specific fields
        db_fields = [
            "cash_balance_amount",
            "cash_balance_currency",
            "total_equity_amount",
            "total_equity_currency",
            "buying_power_amount",
            "buying_power_currency",
            "maintenance_margin_amount",
            "maintenance_margin_currency",
            "day_trading_buying_power_amount",
            "day_trading_buying_power_currency",
            "daily_loss_limit_amount",
            "daily_loss_limit_currency",
        ]
        for field in db_fields:
            data.pop(field, None)

        return Account.model_validate(data)

    async def find_by_name(self, name: str) -> Optional[Account]:
        """Find account by name."""
        query = "SELECT * FROM accounts WHERE name = ?"
        result = await self.db.execute(query, (name,), fetch="one")

        if result:
            table_info = await self.db.get_table_info(self.table_name)
            column_names = [col["name"] for col in table_info]
            data = dict(zip(column_names, result))
            return self._deserialize_entity(data)

        return None

    async def find_active_accounts(self) -> List[Account]:
        """Find all active accounts."""
        query = "SELECT * FROM accounts WHERE is_active = TRUE ORDER BY name"
        results = await self.db.execute(query, fetch="all")

        if results:
            table_info = await self.db.get_table_info(self.table_name)
            column_names = [col["name"] for col in table_info]

            return [
                self._deserialize_entity(dict(zip(column_names, row)))
                for row in results
            ]

        return []


class PortfolioRepository(BaseRepository[Portfolio]):
    """Repository for Portfolio entities."""

    def __init__(self, db_manager: DatabaseManager):
        super().__init__(db_manager, Portfolio, "portfolios")

    def _serialize_entity(self, entity: Portfolio) -> Dict[str, Any]:
        """Serialize Portfolio entity for database storage."""
        return {
            "id": entity.id,
            "created_at": entity.created_at.isoformat(),
            "updated_at": entity.updated_at.isoformat(),
            "version": entity.version,
            "name": entity.name,
            "account_id": entity.account_id,
            "base_currency": entity.base_currency.value,
            "cash_balance_amount": float(entity.cash_balance.amount),
            "cash_balance_currency": entity.cash_balance.currency.value,
            "total_value_amount": float(entity.total_value.amount),
            "total_value_currency": entity.total_value.currency.value,
            "unrealized_pnl_amount": float(entity.unrealized_pnl.amount),
            "unrealized_pnl_currency": entity.unrealized_pnl.currency.value,
            "realized_pnl_amount": float(entity.realized_pnl.amount),
            "realized_pnl_currency": entity.realized_pnl.currency.value,
            "daily_pnl_amount": float(entity.daily_pnl.amount),
            "daily_pnl_currency": entity.daily_pnl.currency.value,
            "max_drawdown": float(entity.max_drawdown),
            "risk_metrics": json.dumps(entity.risk_metrics),
            "performance_metrics": json.dumps(entity.performance_metrics),
        }

    def _deserialize_entity(self, data: Dict[str, Any]) -> Portfolio:
        """Deserialize database data to Portfolio entity."""
        # Convert timestamp strings
        if isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(
                data["created_at"].replace("Z", "+00:00")
            )
        if isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(
                data["updated_at"].replace("Z", "+00:00")
            )

        # Convert Money fields
        data["base_currency"] = Currency(data["base_currency"])
        money_fields = [
            "cash_balance",
            "total_value",
            "unrealized_pnl",
            "realized_pnl",
            "daily_pnl",
        ]

        for field in money_fields:
            data[field] = Money(
                amount=Decimal(str(data[f"{field}_amount"])),
                currency=Currency(data[f"{field}_currency"]),
            )
            # Remove database-specific fields
            del data[f"{field}_amount"]
            del data[f"{field}_currency"]

        # Convert Decimal fields
        data["max_drawdown"] = Decimal(str(data["max_drawdown"]))

        # Parse JSON fields
        data["risk_metrics"] = (
            json.loads(data["risk_metrics"]) if data["risk_metrics"] else {}
        )
        data["performance_metrics"] = (
            json.loads(data["performance_metrics"])
            if data["performance_metrics"]
            else {}
        )

        # Handle positions - load from portfolio_positions table
        # This is a simplified version - positions are stored as a mapping in the Portfolio model
        data["positions"] = {}  # Will be populated by a separate query

        return Portfolio.model_validate(data)

    async def find_by_account_id(self, account_id: str) -> List[Portfolio]:
        """Find portfolios by account ID."""
        query = "SELECT * FROM portfolios WHERE account_id = ? ORDER BY name"
        results = await self.db.execute(query, (account_id,), fetch="all")

        if results:
            table_info = await self.db.get_table_info(self.table_name)
            column_names = [col["name"] for col in table_info]

            return [
                self._deserialize_entity(dict(zip(column_names, row)))
                for row in results
            ]

        return []


class OrderRepository(BaseRepository[Order]):
    """Repository for Order entities."""

    def __init__(self, db_manager: DatabaseManager):
        super().__init__(db_manager, Order, "orders")

    def _serialize_entity(self, entity: Order) -> Dict[str, Any]:
        """Serialize Order entity for database storage."""
        data = {
            "id": entity.id,
            "created_at": entity.created_at.isoformat(),
            "updated_at": entity.updated_at.isoformat(),
            "version": entity.version,
            "symbol_code": entity.symbol.code,
            "symbol_asset_class": entity.symbol.asset_class,
            "order_type": entity.order_type.value,
            "side": entity.side.value,
            "quantity_amount": float(entity.quantity.amount),
            "time_in_force": entity.time_in_force,
            "status": entity.status.value,
            "filled_quantity_amount": float(entity.filled_quantity.amount),
            "tags": json.dumps(entity.tags),
            "child_order_ids": json.dumps(entity.child_order_ids),
        }

        # Handle optional fields
        if entity.price is not None:
            data["price"] = float(entity.price)
        if entity.stop_price is not None:
            data["stop_price"] = float(entity.stop_price)
        if entity.average_fill_price is not None:
            data["average_fill_price"] = float(entity.average_fill_price)
        if entity.commission is not None:
            data["commission_amount"] = float(entity.commission.amount)
            data["commission_currency"] = entity.commission.currency.value
        if entity.strategy_id is not None:
            data["strategy_id"] = entity.strategy_id
        if entity.parent_order_id is not None:
            data["parent_order_id"] = entity.parent_order_id

        return data

    def _deserialize_entity(self, data: Dict[str, Any]) -> Order:
        """Deserialize database data to Order entity."""
        # Convert timestamp strings
        if isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(
                data["created_at"].replace("Z", "+00:00")
            )
        if isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(
                data["updated_at"].replace("Z", "+00:00")
            )

        # Convert Symbol
        data["symbol"] = Symbol(
            code=data["symbol_code"], asset_class=data["symbol_asset_class"]
        )
        del data["symbol_code"]
        del data["symbol_asset_class"]

        # Convert enums
        data["order_type"] = OrderType(data["order_type"])
        data["side"] = OrderSide(data["side"])
        data["status"] = OrderStatus(data["status"])

        # Convert Quantity
        data["quantity"] = Quantity(amount=Decimal(str(data["quantity_amount"])))
        data["filled_quantity"] = Quantity(
            amount=Decimal(str(data["filled_quantity_amount"]))
        )
        del data["quantity_amount"]
        del data["filled_quantity_amount"]

        # Convert optional Decimal fields
        for field in ["price", "stop_price", "average_fill_price"]:
            if data.get(field) is not None:
                data[field] = Decimal(str(data[field]))

        # Convert optional Money field
        if data.get("commission_amount") is not None:
            data["commission"] = Money(
                amount=Decimal(str(data["commission_amount"])),
                currency=Currency(data["commission_currency"]),
            )
            del data["commission_amount"]
            del data["commission_currency"]

        # Parse JSON fields
        data["tags"] = json.loads(data["tags"]) if data["tags"] else []
        data["child_order_ids"] = (
            json.loads(data["child_order_ids"]) if data["child_order_ids"] else []
        )

        return Order.model_validate(data)

    async def find_by_symbol(self, symbol: str) -> List[Order]:
        """Find orders by symbol."""
        query = "SELECT * FROM orders WHERE symbol_code = ? ORDER BY created_at DESC"
        results = await self.db.execute(query, (symbol,), fetch="all")

        return await self._results_to_entities(results)

    async def find_active_orders(self) -> List[Order]:
        """Find all active orders."""
        query = "SELECT * FROM v_active_orders ORDER BY created_at DESC"
        results = await self.db.execute(query, fetch="all")

        return await self._results_to_entities(results)

    async def find_by_status(self, status: OrderStatus) -> List[Order]:
        """Find orders by status."""
        query = "SELECT * FROM orders WHERE status = ? ORDER BY created_at DESC"
        results = await self.db.execute(query, (status.value,), fetch="all")

        return await self._results_to_entities(results)

    async def _results_to_entities(self, results: List[tuple]) -> List[Order]:
        """Convert database results to Order entities."""
        if not results:
            return []

        table_info = await self.db.get_table_info(self.table_name)
        column_names = [col["name"] for col in table_info]

        return [
            self._deserialize_entity(dict(zip(column_names, row))) for row in results
        ]


class TradeRepository(BaseRepository[Trade]):
    """Repository for Trade entities."""

    def __init__(self, db_manager: DatabaseManager):
        super().__init__(db_manager, Trade, "trades")

    def _serialize_entity(self, entity: Trade) -> Dict[str, Any]:
        """Serialize Trade entity for database storage."""
        data = {
            "id": entity.id,
            "created_at": entity.created_at.isoformat(),
            "updated_at": entity.updated_at.isoformat(),
            "version": entity.version,
            "symbol_code": entity.symbol.code,
            "symbol_asset_class": entity.symbol.asset_class,
            "side": entity.side.value,
            "quantity_amount": float(entity.quantity.amount),
            "price": float(entity.price),
            "timestamp": entity.timestamp.isoformat(),
            "order_id": entity.order_id,
            "tags": json.dumps(entity.tags),
        }

        # Handle optional fields
        if entity.position_id is not None:
            data["position_id"] = entity.position_id
        if entity.commission is not None:
            data["commission_amount"] = float(entity.commission.amount)
            data["commission_currency"] = entity.commission.currency.value
        if entity.fees is not None:
            data["fees_amount"] = float(entity.fees.amount)
            data["fees_currency"] = entity.fees.currency.value
        if entity.exchange is not None:
            data["exchange"] = entity.exchange
        if entity.strategy_id is not None:
            data["strategy_id"] = entity.strategy_id

        return data

    def _deserialize_entity(self, data: Dict[str, Any]) -> Trade:
        """Deserialize database data to Trade entity."""
        # Convert timestamp strings
        timestamp_fields = ["created_at", "updated_at", "timestamp"]
        for field in timestamp_fields:
            if isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field].replace("Z", "+00:00"))

        # Convert Symbol
        data["symbol"] = Symbol(
            code=data["symbol_code"], asset_class=data["symbol_asset_class"]
        )
        del data["symbol_code"]
        del data["symbol_asset_class"]

        # Convert enums
        data["side"] = OrderSide(data["side"])

        # Convert Quantity and price
        data["quantity"] = Quantity(amount=Decimal(str(data["quantity_amount"])))
        data["price"] = Decimal(str(data["price"]))
        del data["quantity_amount"]

        # Convert optional Money fields
        for money_field in ["commission", "fees"]:
            amount_key = f"{money_field}_amount"
            currency_key = f"{money_field}_currency"

            if data.get(amount_key) is not None:
                data[money_field] = Money(
                    amount=Decimal(str(data[amount_key])),
                    currency=Currency(data[currency_key]),
                )
                del data[amount_key]
                del data[currency_key]

        # Parse JSON fields
        data["tags"] = json.loads(data["tags"]) if data["tags"] else []

        return Trade.model_validate(data)

    async def find_by_symbol(self, symbol: str) -> List[Trade]:
        """Find trades by symbol."""
        query = "SELECT * FROM trades WHERE symbol_code = ? ORDER BY timestamp DESC"
        results = await self.db.execute(query, (symbol,), fetch="all")

        return await self._results_to_entities(results)

    async def find_by_order_id(self, order_id: str) -> List[Trade]:
        """Find trades by order ID."""
        query = "SELECT * FROM trades WHERE order_id = ? ORDER BY timestamp DESC"
        results = await self.db.execute(query, (order_id,), fetch="all")

        return await self._results_to_entities(results)

    async def find_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[Trade]:
        """Find trades within date range."""
        query = "SELECT * FROM trades WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp DESC"
        results = await self.db.execute(
            query, (start_date.isoformat(), end_date.isoformat()), fetch="all"
        )

        return await self._results_to_entities(results)

    async def _results_to_entities(self, results: List[tuple]) -> List[Trade]:
        """Convert database results to Trade entities."""
        if not results:
            return []

        table_info = await self.db.get_table_info(self.table_name)
        column_names = [col["name"] for col in table_info]

        return [
            self._deserialize_entity(dict(zip(column_names, row))) for row in results
        ]


class PositionRepository(BaseRepository[Position]):
    """Repository for Position entities."""

    def __init__(self, db_manager: DatabaseManager):
        super().__init__(db_manager, Position, "positions")

    def _serialize_entity(self, entity: Position) -> Dict[str, Any]:
        """Serialize Position entity for database storage."""
        data = {
            "id": entity.id,
            "created_at": entity.created_at.isoformat(),
            "updated_at": entity.updated_at.isoformat(),
            "version": entity.version,
            "symbol_code": entity.symbol.code,
            "symbol_asset_class": entity.symbol.asset_class,
            "position_type": entity.position_type.value,
            "quantity_amount": float(entity.quantity.amount),
            "entry_price": float(entity.entry_price),
            "entry_timestamp": entity.entry_timestamp.isoformat(),
            "entry_trades": json.dumps(entity.entry_trades),
            "exit_trades": json.dumps(entity.exit_trades),
            "tags": json.dumps(entity.tags),
            "portfolio_id": entity.portfolio_id,
        }

        # Handle optional fields
        if entity.current_price is not None:
            data["current_price"] = float(entity.current_price)
        if entity.exit_timestamp is not None:
            data["exit_timestamp"] = entity.exit_timestamp.isoformat()
        if entity.strategy_id is not None:
            data["strategy_id"] = entity.strategy_id
        if entity.stop_loss is not None:
            data["stop_loss"] = float(entity.stop_loss)
        if entity.take_profit is not None:
            data["take_profit"] = float(entity.take_profit)

        return data

    def _deserialize_entity(self, data: Dict[str, Any]) -> Position:
        """Deserialize database data to Position entity."""
        # Convert timestamp strings
        timestamp_fields = ["created_at", "updated_at", "entry_timestamp"]
        for field in timestamp_fields:
            if isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field].replace("Z", "+00:00"))

        if data.get("exit_timestamp") and isinstance(data["exit_timestamp"], str):
            data["exit_timestamp"] = datetime.fromisoformat(
                data["exit_timestamp"].replace("Z", "+00:00")
            )

        # Convert Symbol
        data["symbol"] = Symbol(
            code=data["symbol_code"], asset_class=data["symbol_asset_class"]
        )
        del data["symbol_code"]
        del data["symbol_asset_class"]

        # Convert enums
        data["position_type"] = PositionType(data["position_type"])

        # Convert Quantity and prices
        data["quantity"] = Quantity(amount=Decimal(str(data["quantity_amount"])))
        data["entry_price"] = Decimal(str(data["entry_price"]))
        del data["quantity_amount"]

        # Convert optional Decimal fields
        for field in ["current_price", "stop_loss", "take_profit"]:
            if data.get(field) is not None:
                data[field] = Decimal(str(data[field]))

        # Parse JSON fields
        data["entry_trades"] = (
            json.loads(data["entry_trades"]) if data["entry_trades"] else []
        )
        data["exit_trades"] = (
            json.loads(data["exit_trades"]) if data["exit_trades"] else []
        )
        data["tags"] = json.loads(data["tags"]) if data["tags"] else []

        return Position.model_validate(data)

    async def find_open_positions(self) -> List[Position]:
        """Find all open positions."""
        query = "SELECT * FROM v_open_positions ORDER BY entry_timestamp DESC"
        results = await self.db.execute(query, fetch="all")

        return await self._results_to_entities(results)

    async def find_by_portfolio_id(self, portfolio_id: str) -> List[Position]:
        """Find positions by portfolio ID."""
        query = "SELECT * FROM positions WHERE portfolio_id = ? ORDER BY entry_timestamp DESC"
        results = await self.db.execute(query, (portfolio_id,), fetch="all")

        return await self._results_to_entities(results)

    async def find_by_symbol(self, symbol: str) -> List[Position]:
        """Find positions by symbol."""
        query = "SELECT * FROM positions WHERE symbol_code = ? ORDER BY entry_timestamp DESC"
        results = await self.db.execute(query, (symbol,), fetch="all")

        return await self._results_to_entities(results)

    async def _results_to_entities(self, results: List[tuple]) -> List[Position]:
        """Convert database results to Position entities."""
        if not results:
            return []

        table_info = await self.db.get_table_info(self.table_name)
        column_names = [col["name"] for col in table_info]

        return [
            self._deserialize_entity(dict(zip(column_names, row))) for row in results
        ]


class AuditLogRepository:
    """Repository for audit log entries."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.table_name = "audit_logs"

    async def create_log(
        self,
        action: str,
        entity_type: str,
        entity_id: str,
        new_values: Dict[str, Any],
        old_values: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create an audit log entry.

        Args:
            action: Action performed (e.g., 'order_created', 'trade_executed')
            entity_type: Type of entity (e.g., 'order', 'trade', 'position')
            entity_id: ID of the affected entity
            new_values: New values after the action
            old_values: Previous values before the action (for updates)
            user_id: ID of the user who performed the action
            strategy_id: ID of the strategy that performed the action
            metadata: Additional metadata

        Returns:
            ID of the created audit log entry
        """
        import uuid

        log_id = str(uuid.uuid4())

        data = {
            "id": log_id,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": action,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "old_values": json.dumps(old_values) if old_values else None,
            "new_values": json.dumps(new_values, default=str),
            "strategy_id": strategy_id,
            "metadata": json.dumps(metadata or {}),
        }

        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        query = f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})"

        await self.db.execute(query, tuple(data.values()))

        return log_id

    async def find_by_entity(
        self, entity_type: str, entity_id: str, limit: Optional[int] = 100
    ) -> List[Dict[str, Any]]:
        """Find audit logs for a specific entity."""
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE entity_type = ? AND entity_id = ?
            ORDER BY timestamp DESC
        """

        if limit:
            query += f" LIMIT {limit}"

        results = await self.db.execute(query, (entity_type, entity_id), fetch="all")

        if results:
            table_info = await self.db.get_table_info(self.table_name)
            column_names = [col["name"] for col in table_info]

            logs = []
            for row in results:
                log_data = dict(zip(column_names, row))
                # Parse JSON fields
                log_data["old_values"] = (
                    json.loads(log_data["old_values"])
                    if log_data["old_values"]
                    else None
                )
                log_data["new_values"] = (
                    json.loads(log_data["new_values"]) if log_data["new_values"] else {}
                )
                log_data["metadata"] = (
                    json.loads(log_data["metadata"]) if log_data["metadata"] else {}
                )
                logs.append(log_data)

            return logs

        return []

    async def find_by_action(
        self,
        action: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = 100,
    ) -> List[Dict[str, Any]]:
        """Find audit logs by action type and optional date range."""
        conditions = ["action = ?"]
        params = [action]

        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date.isoformat())

        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date.isoformat())

        query = f"""
            SELECT * FROM {self.table_name}
            WHERE {' AND '.join(conditions)}
            ORDER BY timestamp DESC
        """

        if limit:
            query += f" LIMIT {limit}"

        results = await self.db.execute(query, tuple(params), fetch="all")

        if results:
            table_info = await self.db.get_table_info(self.table_name)
            column_names = [col["name"] for col in table_info]

            logs = []
            for row in results:
                log_data = dict(zip(column_names, row))
                # Parse JSON fields
                log_data["old_values"] = (
                    json.loads(log_data["old_values"])
                    if log_data["old_values"]
                    else None
                )
                log_data["new_values"] = (
                    json.loads(log_data["new_values"]) if log_data["new_values"] else {}
                )
                log_data["metadata"] = (
                    json.loads(log_data["metadata"]) if log_data["metadata"] else {}
                )
                logs.append(log_data)

            return logs

        return []
