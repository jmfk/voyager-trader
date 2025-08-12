"""
Tests for SQLite error handling utilities.

This module tests the SQLiteErrorHandler class to ensure robust error code
based error handling instead of fragile string-based checking.
"""

import sqlite3
from unittest.mock import patch

import pytest

from src.voyager_trader.persistence.error_handling import (
    SQLiteErrorCode,
    SQLiteErrorHandler,
)


class TestSQLiteErrorHandler:
    """Test SQLiteErrorHandler functionality."""

    def test_get_error_code_with_error_code(self):
        """Test getting error code from exception with sqlite_errorcode attribute."""
        error = sqlite3.Error("Test error")
        error.sqlite_errorcode = SQLiteErrorCode.SQLITE_CONSTRAINT

        result = SQLiteErrorHandler.get_error_code(error)
        assert result == SQLiteErrorCode.SQLITE_CONSTRAINT

    def test_get_error_code_without_error_code(self):
        """Test getting error code from exception without sqlite_errorcode attribute."""
        error = sqlite3.Error("Test error")

        result = SQLiteErrorHandler.get_error_code(error)
        assert result is None

    def test_is_table_exists_error_positive(self):
        """Test detecting table already exists error."""
        error = sqlite3.Error("table 'users' already exists")
        error.sqlite_errorcode = SQLiteErrorCode.SQLITE_ERROR

        result = SQLiteErrorHandler.is_table_exists_error(error)
        assert result is True

    def test_is_table_exists_error_wrong_code(self):
        """Test table exists check with wrong error code."""
        error = sqlite3.Error("table 'users' already exists")
        error.sqlite_errorcode = SQLiteErrorCode.SQLITE_CONSTRAINT

        result = SQLiteErrorHandler.is_table_exists_error(error)
        assert result is False

    def test_is_table_exists_error_wrong_message(self):
        """Test table exists check with wrong message."""
        error = sqlite3.Error("constraint violation")
        error.sqlite_errorcode = SQLiteErrorCode.SQLITE_ERROR

        result = SQLiteErrorHandler.is_table_exists_error(error)
        assert result is False

    def test_is_constraint_violation_positive(self):
        """Test detecting constraint violation error."""
        error = sqlite3.Error("UNIQUE constraint failed")
        error.sqlite_errorcode = SQLiteErrorCode.SQLITE_CONSTRAINT

        result = SQLiteErrorHandler.is_constraint_violation(error)
        assert result is True

    def test_is_constraint_violation_negative(self):
        """Test constraint violation check with different error code."""
        error = sqlite3.Error("Database error")
        error.sqlite_errorcode = SQLiteErrorCode.SQLITE_ERROR

        result = SQLiteErrorHandler.is_constraint_violation(error)
        assert result is False

    def test_is_database_busy_sqlite_busy(self):
        """Test detecting SQLITE_BUSY error."""
        error = sqlite3.Error("database is locked")
        error.sqlite_errorcode = SQLiteErrorCode.SQLITE_BUSY

        result = SQLiteErrorHandler.is_database_busy(error)
        assert result is True

    def test_is_database_busy_sqlite_locked(self):
        """Test detecting SQLITE_LOCKED error."""
        error = sqlite3.Error("database table is locked")
        error.sqlite_errorcode = SQLiteErrorCode.SQLITE_LOCKED

        result = SQLiteErrorHandler.is_database_busy(error)
        assert result is True

    def test_is_database_busy_negative(self):
        """Test database busy check with different error code."""
        error = sqlite3.Error("Database error")
        error.sqlite_errorcode = SQLiteErrorCode.SQLITE_ERROR

        result = SQLiteErrorHandler.is_database_busy(error)
        assert result is False

    def test_is_database_full_positive(self):
        """Test detecting database full error."""
        error = sqlite3.Error("database or disk is full")
        error.sqlite_errorcode = SQLiteErrorCode.SQLITE_FULL

        result = SQLiteErrorHandler.is_database_full(error)
        assert result is True

    def test_is_database_full_negative(self):
        """Test database full check with different error code."""
        error = sqlite3.Error("Database error")
        error.sqlite_errorcode = SQLiteErrorCode.SQLITE_ERROR

        result = SQLiteErrorHandler.is_database_full(error)
        assert result is False

    def test_is_readonly_error_positive(self):
        """Test detecting readonly database error."""
        error = sqlite3.Error("attempt to write a readonly database")
        error.sqlite_errorcode = SQLiteErrorCode.SQLITE_READONLY

        result = SQLiteErrorHandler.is_readonly_error(error)
        assert result is True

    def test_is_readonly_error_negative(self):
        """Test readonly check with different error code."""
        error = sqlite3.Error("Database error")
        error.sqlite_errorcode = SQLiteErrorCode.SQLITE_ERROR

        result = SQLiteErrorHandler.is_readonly_error(error)
        assert result is False

    def test_handle_database_error_table_exists_ignored(self):
        """Test handling table already exists error when ignored."""
        error = sqlite3.Error("table 'users' already exists")
        error.sqlite_errorcode = SQLiteErrorCode.SQLITE_ERROR

        should_continue, message = SQLiteErrorHandler.handle_database_error(
            error, "test operation", ignore_table_exists=True
        )

        assert should_continue is True
        assert "already exists" in message

    def test_handle_database_error_table_exists_not_ignored(self):
        """Test handling table already exists error when not ignored."""
        error = sqlite3.Error("table 'users' already exists")
        error.sqlite_errorcode = SQLiteErrorCode.SQLITE_ERROR

        should_continue, message = SQLiteErrorHandler.handle_database_error(
            error, "test operation", ignore_table_exists=False
        )

        assert should_continue is False
        assert "test operation" in message

    def test_handle_database_error_busy(self):
        """Test handling database busy error."""
        error = sqlite3.Error("database is locked")
        error.sqlite_errorcode = SQLiteErrorCode.SQLITE_BUSY

        should_continue, message = SQLiteErrorHandler.handle_database_error(
            error, "test operation"
        )

        assert should_continue is False
        assert "busy" in message.lower()
        assert "code 5" in message

    def test_handle_database_error_constraint(self):
        """Test handling constraint violation error."""
        error = sqlite3.Error("UNIQUE constraint failed")
        error.sqlite_errorcode = SQLiteErrorCode.SQLITE_CONSTRAINT

        should_continue, message = SQLiteErrorHandler.handle_database_error(
            error, "test operation"
        )

        assert should_continue is False
        assert "constraint" in message.lower()
        assert "code 19" in message

    def test_handle_database_error_no_error_code(self):
        """Test handling error without error code."""
        error = sqlite3.Error("Generic database error")

        should_continue, message = SQLiteErrorHandler.handle_database_error(
            error, "test operation"
        )

        assert should_continue is False
        assert "test operation" in message
        assert "Generic database error" in message

    @patch("src.voyager_trader.persistence.error_handling.logger")
    def test_log_database_error(self, mock_logger):
        """Test logging database error."""
        error = sqlite3.Error("test error")
        error.sqlite_errorcode = SQLiteErrorCode.SQLITE_CONSTRAINT

        SQLiteErrorHandler.log_database_error(error, "test operation")

        mock_logger.log.assert_called_once()
        args, kwargs = mock_logger.log.call_args
        assert args[0] == 40  # ERROR level
        assert "constraint" in args[1].lower()


class TestSQLiteErrorCode:
    """Test SQLiteErrorCode enum."""

    def test_error_code_values(self):
        """Test that error codes have correct values."""
        assert SQLiteErrorCode.SQLITE_OK == 0
        assert SQLiteErrorCode.SQLITE_ERROR == 1
        assert SQLiteErrorCode.SQLITE_BUSY == 5
        assert SQLiteErrorCode.SQLITE_CONSTRAINT == 19
        assert SQLiteErrorCode.SQLITE_READONLY == 8
        assert SQLiteErrorCode.SQLITE_FULL == 13

    def test_error_code_comparison(self):
        """Test error code comparison operations."""
        assert SQLiteErrorCode.SQLITE_CONSTRAINT > SQLiteErrorCode.SQLITE_ERROR
        assert SQLiteErrorCode.SQLITE_OK < SQLiteErrorCode.SQLITE_ERROR


class TestErrorHandlerIntegration:
    """Integration tests for error handler with actual SQLite operations."""

    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create temporary database path."""
        return str(tmp_path / "test.db")

    def test_table_already_exists_detection(self, temp_db_path):
        """Test detection of table already exists error in real scenario."""
        # Create table first time
        conn = sqlite3.connect(temp_db_path)
        conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY)")
        conn.close()

        # Try to create same table again
        conn = sqlite3.connect(temp_db_path)
        try:
            conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY)")
            assert False, "Expected sqlite3.Error"
        except sqlite3.Error as e:
            is_table_exists = SQLiteErrorHandler.is_table_exists_error(e)
            assert is_table_exists is True
        finally:
            conn.close()

    def test_constraint_violation_detection(self, temp_db_path):
        """Test detection of constraint violation in real scenario."""
        # Create table with unique constraint
        conn = sqlite3.connect(temp_db_path)
        conn.execute(
            "CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT UNIQUE)"
        )
        conn.execute("INSERT INTO test_table (name) VALUES ('test')")

        # Try to insert duplicate
        try:
            conn.execute("INSERT INTO test_table (name) VALUES ('test')")
            assert False, "Expected sqlite3.Error"
        except sqlite3.IntegrityError as e:
            # IntegrityError is a subclass of sqlite3.Error
            is_constraint = SQLiteErrorHandler.is_constraint_violation(e)
            assert is_constraint is True
        finally:
            conn.close()
