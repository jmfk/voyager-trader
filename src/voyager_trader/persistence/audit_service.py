"""
Comprehensive audit logging service for VOYAGER Trader.

This module provides audit logging capabilities that can be used throughout
the trading system to track all actions, changes, and system events.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models.base import BaseEntity
from .database import DatabaseManager, get_database
from .repositories import AuditLogRepository

logger = logging.getLogger(__name__)


class AuditService:
    """
    Comprehensive audit logging service.

    Provides centralized audit logging for all system activities including
    trading operations, user actions, system events, and strategy executions.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize audit service.

        Args:
            db_manager: Database manager instance (will create default if None)
        """
        self.db_manager = db_manager
        self._audit_repo: Optional[AuditLogRepository] = None

    async def _ensure_repository(self) -> None:
        """Ensure audit repository is initialized."""
        if self._audit_repo is None:
            if self.db_manager is None:
                self.db_manager = await get_database()

            self._audit_repo = AuditLogRepository(self.db_manager)

    # Core audit logging methods
    async def log_entity_created(
        self,
        entity: BaseEntity,
        user_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log entity creation.

        Args:
            entity: Created entity
            user_id: ID of the user who created the entity
            strategy_id: ID of the strategy that created the entity
            metadata: Additional metadata

        Returns:
            ID of the audit log entry
        """
        await self._ensure_repository()

        entity_type = entity.__class__.__name__.lower()

        return await self._audit_repo.create_log(
            action=f"{entity_type}_created",
            entity_type=entity_type,
            entity_id=entity.id,
            new_values=entity.model_dump(mode="json"),
            user_id=user_id,
            strategy_id=strategy_id,
            metadata=metadata or {},
        )

    async def log_entity_updated(
        self,
        old_entity: BaseEntity,
        new_entity: BaseEntity,
        user_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log entity update.

        Args:
            old_entity: Entity state before update
            new_entity: Entity state after update
            user_id: ID of the user who updated the entity
            strategy_id: ID of the strategy that updated the entity
            metadata: Additional metadata

        Returns:
            ID of the audit log entry
        """
        await self._ensure_repository()

        entity_type = new_entity.__class__.__name__.lower()

        return await self._audit_repo.create_log(
            action=f"{entity_type}_updated",
            entity_type=entity_type,
            entity_id=new_entity.id,
            old_values=old_entity.model_dump(mode="json"),
            new_values=new_entity.model_dump(mode="json"),
            user_id=user_id,
            strategy_id=strategy_id,
            metadata=metadata or {},
        )

    async def log_entity_deleted(
        self,
        entity: BaseEntity,
        user_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log entity deletion.

        Args:
            entity: Deleted entity
            user_id: ID of the user who deleted the entity
            strategy_id: ID of the strategy that deleted the entity
            metadata: Additional metadata

        Returns:
            ID of the audit log entry
        """
        await self._ensure_repository()

        entity_type = entity.__class__.__name__.lower()

        return await self._audit_repo.create_log(
            action=f"{entity_type}_deleted",
            entity_type=entity_type,
            entity_id=entity.id,
            old_values=entity.model_dump(mode="json"),
            new_values={},
            user_id=user_id,
            strategy_id=strategy_id,
            metadata=metadata or {},
        )

    # Trading-specific audit methods
    async def log_order_submitted(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: Optional[float] = None,
        strategy_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Log order submission to broker."""
        await self._ensure_repository()

        metadata = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_type": order_type,
            "price": price,
        }

        return await self._audit_repo.create_log(
            action="order_submitted_to_broker",
            entity_type="order",
            entity_id=order_id,
            new_values=metadata,
            user_id=user_id,
            strategy_id=strategy_id,
            metadata=metadata,
        )

    async def log_order_filled(
        self,
        order_id: str,
        symbol: str,
        quantity: float,
        fill_price: float,
        commission: Optional[float] = None,
        strategy_id: Optional[str] = None,
    ) -> str:
        """Log order fill/execution."""
        await self._ensure_repository()

        metadata = {
            "symbol": symbol,
            "filled_quantity": quantity,
            "fill_price": fill_price,
            "commission": commission,
            "notional_value": quantity * fill_price,
        }

        return await self._audit_repo.create_log(
            action="order_filled",
            entity_type="order",
            entity_id=order_id,
            new_values=metadata,
            strategy_id=strategy_id,
            metadata=metadata,
        )

    async def log_order_cancelled(
        self,
        order_id: str,
        reason: str,
        strategy_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Log order cancellation."""
        await self._ensure_repository()

        metadata = {"cancellation_reason": reason}

        return await self._audit_repo.create_log(
            action="order_cancelled",
            entity_type="order",
            entity_id=order_id,
            new_values=metadata,
            user_id=user_id,
            strategy_id=strategy_id,
            metadata=metadata,
        )

    async def log_position_opened(
        self,
        position_id: str,
        symbol: str,
        position_type: str,
        quantity: float,
        entry_price: float,
        cost_basis: float,
        strategy_id: Optional[str] = None,
    ) -> str:
        """Log position opening."""
        await self._ensure_repository()

        metadata = {
            "symbol": symbol,
            "position_type": position_type,
            "quantity": quantity,
            "entry_price": entry_price,
            "cost_basis": cost_basis,
        }

        return await self._audit_repo.create_log(
            action="position_opened",
            entity_type="position",
            entity_id=position_id,
            new_values=metadata,
            strategy_id=strategy_id,
            metadata=metadata,
        )

    async def log_position_closed(
        self,
        position_id: str,
        symbol: str,
        exit_price: float,
        realized_pnl: float,
        holding_period_days: float,
        strategy_id: Optional[str] = None,
    ) -> str:
        """Log position closing."""
        await self._ensure_repository()

        metadata = {
            "symbol": symbol,
            "exit_price": exit_price,
            "realized_pnl": realized_pnl,
            "holding_period_days": holding_period_days,
        }

        return await self._audit_repo.create_log(
            action="position_closed",
            entity_type="position",
            entity_id=position_id,
            new_values=metadata,
            strategy_id=strategy_id,
            metadata=metadata,
        )

    # System event audit methods
    async def log_system_startup(
        self,
        version: str,
        config: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> str:
        """Log system startup."""
        await self._ensure_repository()

        metadata = {
            "system_version": version,
            "startup_time": datetime.utcnow().isoformat(),
            "configuration": config,
        }

        return await self._audit_repo.create_log(
            action="system_startup",
            entity_type="system",
            entity_id="voyager_trader",
            new_values=metadata,
            user_id=user_id,
            metadata=metadata,
        )

    async def log_system_shutdown(
        self,
        reason: str = "normal_shutdown",
        user_id: Optional[str] = None,
    ) -> str:
        """Log system shutdown."""
        await self._ensure_repository()

        metadata = {
            "shutdown_reason": reason,
            "shutdown_time": datetime.utcnow().isoformat(),
        }

        return await self._audit_repo.create_log(
            action="system_shutdown",
            entity_type="system",
            entity_id="voyager_trader",
            new_values=metadata,
            user_id=user_id,
            metadata=metadata,
        )

    async def log_strategy_started(
        self,
        strategy_id: str,
        strategy_name: str,
        parameters: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> str:
        """Log strategy start."""
        await self._ensure_repository()

        metadata = {
            "strategy_name": strategy_name,
            "parameters": parameters,
            "start_time": datetime.utcnow().isoformat(),
        }

        return await self._audit_repo.create_log(
            action="strategy_started",
            entity_type="strategy",
            entity_id=strategy_id,
            new_values=metadata,
            user_id=user_id,
            strategy_id=strategy_id,
            metadata=metadata,
        )

    async def log_strategy_stopped(
        self,
        strategy_id: str,
        reason: str,
        performance_metrics: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Log strategy stop."""
        await self._ensure_repository()

        metadata = {
            "stop_reason": reason,
            "stop_time": datetime.utcnow().isoformat(),
            "performance_metrics": performance_metrics or {},
        }

        return await self._audit_repo.create_log(
            action="strategy_stopped",
            entity_type="strategy",
            entity_id=strategy_id,
            new_values=metadata,
            user_id=user_id,
            strategy_id=strategy_id,
            metadata=metadata,
        )

    async def log_error(
        self,
        error_type: str,
        error_message: str,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        stack_trace: Optional[str] = None,
        user_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
    ) -> str:
        """Log system or strategy errors."""
        await self._ensure_repository()

        metadata = {
            "error_type": error_type,
            "error_message": error_message,
            "stack_trace": stack_trace,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return await self._audit_repo.create_log(
            action="error_occurred",
            entity_type=entity_type or "system",
            entity_id=entity_id or "unknown",
            new_values=metadata,
            user_id=user_id,
            strategy_id=strategy_id,
            metadata=metadata,
        )

    # Risk management audit methods
    async def log_risk_limit_breached(
        self,
        limit_type: str,
        limit_value: float,
        current_value: float,
        entity_type: str,
        entity_id: str,
        action_taken: str,
        strategy_id: Optional[str] = None,
    ) -> str:
        """Log risk limit breach."""
        await self._ensure_repository()

        metadata = {
            "limit_type": limit_type,
            "limit_value": limit_value,
            "current_value": current_value,
            "action_taken": action_taken,
            "breach_time": datetime.utcnow().isoformat(),
        }

        return await self._audit_repo.create_log(
            action="risk_limit_breached",
            entity_type=entity_type,
            entity_id=entity_id,
            new_values=metadata,
            strategy_id=strategy_id,
            metadata=metadata,
        )

    async def log_circuit_breaker_triggered(
        self,
        breaker_type: str,
        trigger_condition: str,
        entity_type: str,
        entity_id: str,
        strategy_id: Optional[str] = None,
    ) -> str:
        """Log circuit breaker activation."""
        await self._ensure_repository()

        metadata = {
            "breaker_type": breaker_type,
            "trigger_condition": trigger_condition,
            "trigger_time": datetime.utcnow().isoformat(),
        }

        return await self._audit_repo.create_log(
            action="circuit_breaker_triggered",
            entity_type=entity_type,
            entity_id=entity_id,
            new_values=metadata,
            strategy_id=strategy_id,
            metadata=metadata,
        )

    # User action audit methods
    async def log_user_login(
        self,
        user_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Log user login."""
        await self._ensure_repository()

        metadata = {
            "login_time": datetime.utcnow().isoformat(),
            "ip_address": ip_address,
            "user_agent": user_agent,
            "session_id": session_id,
        }

        return await self._audit_repo.create_log(
            action="user_login",
            entity_type="user",
            entity_id=user_id,
            new_values=metadata,
            user_id=user_id,
            metadata=metadata,
        )

    async def log_user_logout(
        self,
        user_id: str,
        session_duration_minutes: Optional[float] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Log user logout."""
        await self._ensure_repository()

        metadata = {
            "logout_time": datetime.utcnow().isoformat(),
            "session_duration_minutes": session_duration_minutes,
            "session_id": session_id,
        }

        return await self._audit_repo.create_log(
            action="user_logout",
            entity_type="user",
            entity_id=user_id,
            new_values=metadata,
            user_id=user_id,
            metadata=metadata,
        )

    # Configuration audit methods
    async def log_configuration_changed(
        self,
        config_key: str,
        old_value: Any,
        new_value: Any,
        user_id: Optional[str] = None,
    ) -> str:
        """Log configuration changes."""
        await self._ensure_repository()

        metadata = {
            "config_key": config_key,
            "change_time": datetime.utcnow().isoformat(),
        }

        return await self._audit_repo.create_log(
            action="configuration_changed",
            entity_type="configuration",
            entity_id=config_key,
            old_values={"value": old_value},
            new_values={"value": new_value},
            user_id=user_id,
            metadata=metadata,
        )

    # Query methods
    async def get_audit_trail(
        self,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        action: Optional[str] = None,
        user_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get filtered audit trail.

        Args:
            entity_type: Filter by entity type
            entity_id: Filter by entity ID
            action: Filter by action
            user_id: Filter by user ID
            strategy_id: Filter by strategy ID
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of results

        Returns:
            List of audit log entries
        """
        await self._ensure_repository()

        if entity_type and entity_id:
            return await self._audit_repo.find_by_entity(entity_type, entity_id, limit)
        elif action:
            return await self._audit_repo.find_by_action(
                action, start_date, end_date, limit
            )
        else:
            # For more complex queries, we'd need to extend the repository
            # For now, return recent logs
            return await self._audit_repo.find_by_action(
                "*", start_date, end_date, limit
            )

    async def get_entity_history(
        self,
        entity_type: str,
        entity_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get complete history of an entity."""
        await self._ensure_repository()

        return await self._audit_repo.find_by_entity(entity_type, entity_id, limit)

    async def get_user_activity(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get user activity logs."""
        await self._ensure_repository()

        # This would require a more sophisticated query in the repository
        # For now, we'll return a placeholder
        return []

    async def get_strategy_activity(
        self,
        strategy_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get strategy activity logs."""
        await self._ensure_repository()

        # This would require a more sophisticated query in the repository
        # For now, we'll return a placeholder
        return []


# Global audit service instance
_audit_service: Optional[AuditService] = None


async def get_audit_service() -> AuditService:
    """
    Get the global audit service instance.

    Returns:
        Initialized audit service
    """
    global _audit_service
    if _audit_service is None:
        _audit_service = AuditService()

    return _audit_service


# Convenience functions for common audit operations
async def audit_entity_created(entity: BaseEntity, **kwargs) -> str:
    """Convenience function to audit entity creation."""
    service = await get_audit_service()
    return await service.log_entity_created(entity, **kwargs)


async def audit_entity_updated(
    old_entity: BaseEntity, new_entity: BaseEntity, **kwargs
) -> str:
    """Convenience function to audit entity updates."""
    service = await get_audit_service()
    return await service.log_entity_updated(old_entity, new_entity, **kwargs)


async def audit_entity_deleted(entity: BaseEntity, **kwargs) -> str:
    """Convenience function to audit entity deletion."""
    service = await get_audit_service()
    return await service.log_entity_deleted(entity, **kwargs)
