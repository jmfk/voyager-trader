"""
SQLite error handling utilities for VOYAGER Trader.

This module provides robust error handling for SQLite operations using proper
error codes instead of fragile string-based error checking.
"""

import logging
import sqlite3
from enum import IntEnum
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class SQLiteErrorCode(IntEnum):
    """SQLite error codes for reliable error handling."""

    # Core SQLite error codes
    SQLITE_OK = 0  # Successful result
    SQLITE_ERROR = 1  # Generic error
    SQLITE_INTERNAL = 2  # Internal logic error in SQLite
    SQLITE_PERM = 3  # Access permission denied
    SQLITE_ABORT = 4  # Callback routine requested an abort
    SQLITE_BUSY = 5  # Database file is locked
    SQLITE_LOCKED = 6  # Database table is locked
    SQLITE_NOMEM = 7  # Out of memory
    SQLITE_READONLY = 8  # Attempt to write a readonly database
    SQLITE_INTERRUPT = 9  # Operation was interrupted
    SQLITE_IOERR = 10  # Disk I/O error occurred
    SQLITE_CORRUPT = 11  # Database disk image is malformed
    SQLITE_NOTFOUND = 12  # Unknown opcode in sqlite3_file_control()
    SQLITE_FULL = 13  # Insertion failed because database is full
    SQLITE_CANTOPEN = 14  # Unable to open database file
    SQLITE_PROTOCOL = 15  # Database lock protocol error
    SQLITE_EMPTY = 16  # Internal use only
    SQLITE_SCHEMA = 17  # Database schema changed
    SQLITE_TOOBIG = 18  # String or BLOB exceeds size limit
    SQLITE_CONSTRAINT = 19  # Constraint violation

    # Extended error codes (high byte contains additional info)
    SQLITE_CONSTRAINT_UNIQUE = 2067  # UNIQUE constraint failed
    SQLITE_MISMATCH = 20  # Data type mismatch
    SQLITE_MISUSE = 21  # Library used incorrectly
    SQLITE_NOLFS = 22  # Uses OS features not supported on host
    SQLITE_AUTH = 23  # Authorization denied
    SQLITE_FORMAT = 24  # Not used
    SQLITE_RANGE = 25  # 2nd parameter to sqlite3_bind out of range
    SQLITE_NOTADB = 26  # File opened that is not a database file


class SQLiteErrorHandler:
    """Centralized SQLite error handling with proper error codes."""

    @staticmethod
    def get_error_code(error: sqlite3.Error) -> Optional[int]:
        """
        Extract SQLite error code from exception.

        Args:
            error: SQLite exception

        Returns:
            Error code if available, None otherwise
        """
        return getattr(error, "sqlite_errorcode", None)

    @staticmethod
    def is_table_exists_error(error: sqlite3.Error) -> bool:
        """
        Check if error indicates table already exists.

        Args:
            error: SQLite exception

        Returns:
            True if error indicates table already exists
        """
        error_code = SQLiteErrorHandler.get_error_code(error)

        # For table creation, SQLITE_ERROR typically means "already exists"
        if error_code == SQLiteErrorCode.SQLITE_ERROR:
            error_msg = str(error).lower()
            return "table" in error_msg and "already exists" in error_msg

        return False

    @staticmethod
    def is_constraint_violation(error: sqlite3.Error) -> bool:
        """
        Check if error is a constraint violation.

        Args:
            error: SQLite exception

        Returns:
            True if error is a constraint violation
        """
        error_code = SQLiteErrorHandler.get_error_code(error)

        # Check for both primary error code and extended error codes
        if error_code == SQLiteErrorCode.SQLITE_CONSTRAINT:
            return True

        # Check for extended constraint error codes
        if error_code == SQLiteErrorCode.SQLITE_CONSTRAINT_UNIQUE:
            return True

        # For constraint violations, the primary code is stored in the low byte
        if (
            error_code is not None
            and (error_code & 0xFF) == SQLiteErrorCode.SQLITE_CONSTRAINT
        ):
            return True

        return False

    @staticmethod
    def is_database_busy(error: sqlite3.Error) -> bool:
        """
        Check if error indicates database is busy/locked.

        Args:
            error: SQLite exception

        Returns:
            True if database is busy or locked
        """
        error_code = SQLiteErrorHandler.get_error_code(error)
        return error_code in (
            SQLiteErrorCode.SQLITE_BUSY,
            SQLiteErrorCode.SQLITE_LOCKED,
        )

    @staticmethod
    def is_database_full(error: sqlite3.Error) -> bool:
        """
        Check if error indicates database is full.

        Args:
            error: SQLite exception

        Returns:
            True if database is full
        """
        error_code = SQLiteErrorHandler.get_error_code(error)
        return error_code == SQLiteErrorCode.SQLITE_FULL

    @staticmethod
    def is_readonly_error(error: sqlite3.Error) -> bool:
        """
        Check if error indicates readonly database.

        Args:
            error: SQLite exception

        Returns:
            True if database is readonly
        """
        error_code = SQLiteErrorHandler.get_error_code(error)
        return error_code == SQLiteErrorCode.SQLITE_READONLY

    @staticmethod
    def handle_database_error(
        error: sqlite3.Error, operation: str, ignore_table_exists: bool = False
    ) -> Tuple[bool, str]:
        """
        Handle SQLite database errors with proper error codes.

        Args:
            error: SQLite exception
            operation: Description of the operation that failed
            ignore_table_exists: Whether to ignore "table already exists" errors

        Returns:
            Tuple of (should_continue, log_message)
            - should_continue: True if operation should continue, False if re-raise
            - log_message: Message to log
        """
        error_code = SQLiteErrorHandler.get_error_code(error)
        error_msg = str(error)

        if ignore_table_exists and SQLiteErrorHandler.is_table_exists_error(error):
            return True, f"Table already exists during {operation}, continuing"

        if SQLiteErrorHandler.is_database_busy(error):
            return (
                False,
                f"Database busy during {operation} (code {error_code}): {error_msg}",
            )

        if SQLiteErrorHandler.is_constraint_violation(error):
            return (
                False,
                f"Constraint violation during {operation} "
                f"(code {error_code}): {error_msg}",
            )

        if SQLiteErrorHandler.is_database_full(error):
            return (
                False,
                f"Database full during {operation} (code {error_code}): {error_msg}",
            )

        if SQLiteErrorHandler.is_readonly_error(error):
            return (
                False,
                f"Database readonly during {operation} "
                f"(code {error_code}): {error_msg}",
            )

        # Unknown error - log with error code if available
        if error_code is not None:
            return (
                False,
                f"Database error during {operation} (code {error_code}): {error_msg}",
            )
        else:
            return False, f"Database error during {operation}: {error_msg}"

    @staticmethod
    def log_database_error(
        error: sqlite3.Error, operation: str, level: int = logging.ERROR
    ) -> None:
        """
        Log database error with error code information.

        Args:
            error: SQLite exception
            operation: Description of the operation that failed
            level: Logging level
        """
        should_continue, log_message = SQLiteErrorHandler.handle_database_error(
            error, operation, ignore_table_exists=False
        )

        logger.log(level, log_message)
