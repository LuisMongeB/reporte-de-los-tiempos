"""
Tests for database connection and session management.
"""

import pytest
import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import logging
from src.core.database import (
    DatabaseManager,
    DatabaseConnectionError,
    get_database_manager,
    close_database,
    reset_database_manager,
    get_db_session,
    Base
)
from src.core.config import TestConfig, DevelopmentConfig


class TestDatabaseManager:
    """Test DatabaseManager class"""
    
    def setup_method(self):
        """Reset database manager before each test"""
        reset_database_manager()
    
    @pytest.fixture
    def test_config(self):
        """Provide test configuration"""
        return TestConfig()
    
    @pytest.fixture
    def file_config(self):
        """Provide configuration with actual file database"""
        config = TestConfig()
        # Use temporary file for testing
        temp_dir = Path(tempfile.mkdtemp())
        config.database_url = f"sqlite+aiosqlite:///{temp_dir}/test.db"
        return config
    
    @pytest.mark.asyncio
    async def test_database_manager_initialization(self, test_config):
        """Test DatabaseManager initialization"""
        db_manager = DatabaseManager(test_config)
        
        # Should not be initialized yet
        assert not db_manager._initialized
        assert db_manager.engine is None
        assert db_manager.session_factory is None
        
        # Initialize
        await db_manager.initialize()
        
        # Should be initialized now
        assert db_manager._initialized
        assert db_manager.engine is not None
        assert db_manager.session_factory is not None
        
        # Cleanup
        await db_manager.close()
    
    @pytest.mark.asyncio
    async def test_double_initialization_warning(self, test_config, caplog):
        """Test that double initialization logs a warning"""
        db_manager = DatabaseManager(test_config)
        
        await db_manager.initialize()
        await db_manager.initialize()  # Second initialization
        
        assert "Database already initialized" in caplog.text
        
        await db_manager.close()
    
    @pytest.mark.asyncio
    async def test_session_context_manager(self, test_config):
        """Test getting and using database sessions"""
        db_manager = DatabaseManager(test_config)
        await db_manager.initialize()
        
        # Test session context manager
        async with db_manager.get_session() as session:
            assert isinstance(session, AsyncSession)
            # Test simple query
            result = await session.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            assert row[0] == 1
        
        await db_manager.close()
    
    @pytest.mark.asyncio
    async def test_session_without_initialization(self, test_config):
        """Test that getting session without initialization raises error"""
        db_manager = DatabaseManager(test_config)
        
        with pytest.raises(DatabaseConnectionError) as exc_info:
            async with db_manager.get_session() as session:
                pass
        
        assert "Database not initialized" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_session_error_handling(self, test_config):
        """Test that session errors trigger rollback"""
        db_manager = DatabaseManager(test_config)
        await db_manager.initialize()
        
        with pytest.raises(Exception):
            async with db_manager.get_session() as session:
                # Force an error
                await session.execute(text("SELECT * FROM nonexistent_table"))
        
        # Database should still be functional after error
        async with db_manager.get_session() as session:
            result = await session.execute(text("SELECT 1"))
            assert result.fetchone()[0] == 1
        
        await db_manager.close()
    
    @pytest.mark.asyncio
    async def test_connection_health_check(self, test_config):
        """Test database connection health check"""
        db_manager = DatabaseManager(test_config)
        
        # Should return False before initialization
        assert await db_manager.check_connection() is False
        
        # Initialize and test
        await db_manager.initialize()
        assert await db_manager.check_connection() is True
        
        # Close and test
        await db_manager.close()
        assert await db_manager.check_connection() is False
    
    @pytest.mark.asyncio
    async def test_create_database_file_memory(self, test_config, caplog):
        """Test database file creation for in-memory database"""
        caplog.set_level(logging.INFO)
        
        db_manager = DatabaseManager(test_config)
        
        await db_manager.create_database_file()
        
        assert "Using in-memory database" in caplog.text
    
    @pytest.mark.asyncio
    async def test_create_database_file_disk(self, file_config, caplog):
        """Test database file creation for file database"""
        caplog.set_level(logging.INFO)
        db_manager = DatabaseManager(file_config)
        
        # Extract path from config
        db_path_str = file_config.database_url.replace("sqlite+aiosqlite:///", "")
        db_path = Path(db_path_str)
        
        # Ensure file doesn't exist
        if db_path.exists():
            db_path.unlink()
        
        await db_manager.create_database_file()
        
        # File should be created
        assert db_path.exists()
        assert f"Created database file: {db_path}" in caplog.text
        
        # Test when file already exists
        caplog.clear()
        await db_manager.create_database_file()
        assert f"Database file exists: {db_path}" in caplog.text
        
        # Cleanup
        if db_path.exists():
            db_path.unlink()
        db_path.parent.rmdir()
    
    @pytest.mark.asyncio
    async def test_get_raw_connection(self, file_config):
        """Test getting raw aiosqlite connection"""
        db_manager = DatabaseManager(file_config)
        await db_manager.create_database_file()
        await db_manager.initialize()
        
        # Get raw connection
        conn = await db_manager.get_raw_connection()
        
        # Test that it works
        cursor = await conn.execute("SELECT 1")
        result = await cursor.fetchone()
        assert result[0] == 1
        
        await conn.close()
        await db_manager.close()
        
        # Cleanup
        db_path_str = file_config.database_url.replace("sqlite+aiosqlite:///", "")
        db_path = Path(db_path_str)
        if db_path.exists():
            db_path.unlink()
        db_path.parent.rmdir()
    
    @pytest.mark.asyncio
    async def test_get_raw_connection_without_initialization(self, test_config):
        """Test getting raw connection without initialization"""
        db_manager = DatabaseManager(test_config)
        
        with pytest.raises(DatabaseConnectionError) as exc_info:
            await db_manager.get_raw_connection()
        
        assert "Database not initialized" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_close_database(self, test_config):
        """Test closing database connections"""
        db_manager = DatabaseManager(test_config)
        await db_manager.initialize()
        
        assert db_manager._initialized is True
        
        await db_manager.close()
        
        assert db_manager._initialized is False


class TestGlobalDatabaseFunctions:
    """Test global database management functions"""
    
    def setup_method(self):
        """Reset database manager before each test"""
        reset_database_manager()
    
    @pytest.mark.asyncio
    async def test_get_database_manager_with_config(self):
        """Test getting database manager with explicit config"""
        config = TestConfig()
        
        db_manager = await get_database_manager(config)
        
        assert db_manager is not None
        assert db_manager._initialized is True
        assert db_manager.config == config
        
        # Should return same instance on second call
        db_manager2 = await get_database_manager()
        assert db_manager is db_manager2
        
        await close_database()
    
    @pytest.mark.asyncio
    async def test_get_database_manager_without_config(self):
        """Test getting database manager without explicit config"""
        with patch('src.core.config.get_config') as mock_get_config:
            mock_config = TestConfig()
            mock_get_config.return_value = mock_config
            
            db_manager = await get_database_manager()
            
            assert db_manager is not None
            assert db_manager._initialized is True
            mock_get_config.assert_called_once()
        
        await close_database()
    
    @pytest.mark.asyncio
    async def test_close_database_function(self):
        """Test global close database function"""
        config = TestConfig()
        
        # Get database manager
        db_manager = await get_database_manager(config)
        assert db_manager._initialized is True
        
        # Close it
        await close_database()
        
        # Global manager should be reset
        from src.core.database import _db_manager
        assert _db_manager is None
    
    @pytest.mark.asyncio
    async def test_get_db_session_convenience_function(self):
        """Test convenience function for getting database sessions"""
        config = TestConfig()
        
        # Use the convenience function
        async with get_db_session() as session:
            assert isinstance(session, AsyncSession)
            result = await session.execute(text("SELECT 1"))
            assert result.fetchone()[0] == 1
        
        await close_database()
    
    def test_reset_database_manager(self):
        """Test resetting database manager"""
        from src.core.database import _db_manager
        
        # Simulate having a database manager
        with patch('src.core.database._db_manager', 'fake_manager'):
            reset_database_manager()
            # Should be None after reset
            assert _db_manager is None


class TestDatabaseConnectionError:
    """Test DatabaseConnectionError exception"""
    
    def test_database_connection_error(self):
        """Test DatabaseConnectionError can be raised and caught"""
        with pytest.raises(DatabaseConnectionError) as exc_info:
            raise DatabaseConnectionError("Test error message")
        
        assert "Test error message" in str(exc_info.value)


class TestDatabaseIntegration:
    """Integration tests for database functionality"""
    
    def setup_method(self):
        """Reset database manager before each test"""
        reset_database_manager()
    
    @pytest.mark.asyncio
    async def test_full_database_lifecycle(self):
        """Test complete database lifecycle"""
        config = TestConfig()
        
        # 1. Get database manager
        db_manager = await get_database_manager(config)
        assert await db_manager.check_connection() is True
        
        # 2. Use session for operations
        async with get_db_session() as session:
            result = await session.execute(text("SELECT 1 as test"))
            assert result.fetchone()[0] == 1
        
        # 3. Check health
        assert await db_manager.check_connection() is True
        
        # 4. Close everything
        await close_database()
        
        # 5. Verify cleanup
        from src.core.database import _db_manager
        assert _db_manager is None