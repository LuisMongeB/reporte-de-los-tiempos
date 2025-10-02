"""
Database connection and session management for SQLite with async support.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator
from pathlib import Path

import aiosqlite
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text, pool

from src.core.config import BaseConfig

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for all database models"""
    pass


class DatabaseManager:
    """Manages SQLite database connections and sessions"""
    
    def __init__(self, config: BaseConfig):
        self.config = config
        self.engine: Optional[object] = None
        self.session_factory: Optional[async_sessionmaker] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the database engine and session factory"""
        if self._initialized:
            logger.warning("Database already initialized")
            return
        
        try:
            # Create async engine for SQLite
            self.engine = create_async_engine(
                self.config.database_url,
                echo=self.config.database_echo,
                # SQLite specific settings
                poolclass=pool.StaticPool,
                connect_args={
                    "check_same_thread": False,
                    "isolation_level": None,
                },
                # Performance settings
                pool_pre_ping=True,
                pool_recycle=300,
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False,
            )
            
            self._initialized = True
            logger.info(f"Database initialized with URL: {self.config.database_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseConnectionError(f"Database initialization failed: {e}")
    
    async def close(self) -> None:
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")
        self._initialized = False
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get an async database session with automatic cleanup.
        
        Usage:
            async with db_manager.get_session() as session:
                # Use session here
                result = await session.execute(...)
        """
        if not self._initialized:
            raise DatabaseConnectionError("Database not initialized. Call initialize() first.")
        
        if not self.session_factory:
            raise DatabaseConnectionError("Session factory not available")
        
        session = self.session_factory()
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()
    
    async def check_connection(self) -> bool:
        """
        Check if database connection is healthy.
        
        Returns:
            bool: True if connection is healthy, False otherwise
        """
        if not self._initialized or not self.engine:
            return False
        
        try:
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def create_database_file(self) -> None:
        """Create database file and directory if they don't exist"""
        if self.config.database_url.startswith("sqlite+aiosqlite://"):
            # Extract file path from URL
            db_path_str = self.config.database_url.replace("sqlite+aiosqlite:///", "")
            
            # Handle in-memory database
            if db_path_str == ":memory:":
                logger.info("Using in-memory database")
                return
            
            db_path = Path(db_path_str)
            
            # Create directory if it doesn't exist
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create empty database file if it doesn't exist
            if not db_path.exists():
                db_path.touch()
                logger.info(f"Created database file: {db_path}")
            else:
                logger.info(f"Database file exists: {db_path}")
    
    async def create_tables(self) -> None:
        """Create all database tables from models"""
        if not self._initialized:
            raise DatabaseConnectionError("Database not initialized. Call initialize() first.")
        
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise DatabaseConnectionError(f"Table creation failed: {e}")
    
    async def get_raw_connection(self) -> aiosqlite.Connection:
        """
        Get a raw aiosqlite connection for direct SQL operations.
        Use with caution - prefer get_session() for ORM operations.
        """
        if not self._initialized:
            raise DatabaseConnectionError("Database not initialized")
        
        if self.config.database_url.startswith("sqlite+aiosqlite://"):
            db_path = self.config.database_url.replace("sqlite+aiosqlite:///", "")
            return await aiosqlite.connect(db_path)
        else:
            raise DatabaseConnectionError("Raw connections only supported for SQLite")


class DatabaseConnectionError(Exception):
    """Raised when database connection operations fail"""
    pass


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


async def get_database_manager(config: Optional[BaseConfig] = None) -> DatabaseManager:
    """
    Get the global database manager instance.
    
    Args:
        config: Optional configuration. If not provided, will load from get_config()
        
    Returns:
        DatabaseManager: The database manager instance
    """
    global _db_manager
    
    if _db_manager is None:
        if config is None:
            from src.core.config import get_config
            config = get_config()
        
        _db_manager = DatabaseManager(config)
        await _db_manager.create_database_file()
        await _db_manager.initialize()
    
    return _db_manager


async def close_database():
    """Close the global database manager"""
    global _db_manager
    if _db_manager:
        await _db_manager.close()
        _db_manager = None


@asynccontextmanager
def reset_database_manager():
    """Reset the global database manager (useful for testing)"""
    global _db_manager
    _db_manager = None


# Convenience function for getting sessions
@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Convenience function to get a database session.
    
    Usage:
        async with get_db_session() as session:
            # Use session here
    """
    db_manager = await get_database_manager()
    async with db_manager.get_session() as session:
        yield session